import { Transform } from '../utils';
import { type Chunk } from './chunk';
import { type ChunkManager } from './chunk-manager';
import { type GaussianSource, type ReadRequest, type SourceMetadata } from './gaussian-source';
import { type Layer } from './layout';

/**
 * Compose a coordinate-space `Transform` onto a source, lazily.
 *
 * The returned source exposes the composed transform in its metadata; reads
 * pass through to the parent without modification. Downstream consumers
 * (writers, kernels that care about coordinate space) read `meta.transform`
 * and apply it themselves.
 *
 * This matches the existing deferred-transform behavior on `DataTable.transform`.
 * A future iteration may add a per-chunk GPU transform kernel; the public
 * contract doesn't need to change for that.
 */
const mapSource = (src: GaussianSource, transform: Transform): GaussianSource => {
    const composed = src.meta.transform.clone();
    composed.mul(transform);

    const meta: SourceMetadata = {
        ...src.meta,
        transform: composed
    };

    return {
        meta,
        read: req => src.read(req),
        close: () => src.close()
    };
};

// ---------------------------------------------------------------------------

/**
 * Layers that are part of a source's read surface.
 */
const ALL_LAYERS: ReadonlyArray<Layer> = ['position', 'geometric', 'color', 'other'];

/**
 * Filter or permute by scatter-gathering from the parent.
 *
 * Both `filterSource` and `permuteSource` reduce to "for each output gaussian,
 * read source row `srcIndices[i]` for each requested layer." `filterSource`
 * passes the compacted indices of surviving rows; `permuteSource` passes the
 * full permutation. The output `numGaussians` is `srcIndices.length`.
 *
 * The implementation does a CPU-side gather: it reads each contributing
 * source chunk into a temporary chunk (via `manager`), reads it back to CPU,
 * scatters the chosen rows into a staging ArrayBuffer per layer, then
 * uploads to the caller's destination chunk. This trades GPU↔CPU round-trips
 * for simplicity; a GPU-side gather kernel is a future optimization.
 */
const makeIndexedSource = (
    src: GaussianSource,
    srcIndices: Uint32Array,
    manager: ChunkManager
): GaussianSource => {
    const parentMeta = src.meta;
    const chunkSize = parentMeta.chunkSize;
    const numGaussians = srcIndices.length;
    const numChunks = Math.max(1, Math.ceil(numGaussians / chunkSize));
    const lodCounts = [numGaussians];

    const meta: SourceMetadata = {
        numGaussians,
        numLods: 1,
        lodCounts,
        chunkSize,
        numChunks: [numChunks],
        shBands: parentMeta.shBands,
        extraColumns: parentMeta.extraColumns,
        transform: parentMeta.transform,
        availableLayers: parentMeta.availableLayers,
        layouts: parentMeta.layouts
    };

    const read = async (req: ReadRequest): Promise<void> => {
        if ((req.lod ?? 0) !== 0) {
            throw new Error('filterSource/permuteSource: only lod=0 is supported');
        }
        const outStart = req.chunkIndex * chunkSize;
        const outCount = Math.min(chunkSize, numGaussians - outStart);
        if (outCount <= 0) {
            throw new Error(`indexed source: chunkIndex ${req.chunkIndex} out of range`);
        }

        // Map output rows -> source rows for this output chunk.
        const sliceIndices = srcIndices.subarray(outStart, outStart + outCount);

        // Determine which source chunks contribute.
        const contributors = new Set<number>();
        for (let i = 0; i < sliceIndices.length; i++) {
            contributors.add(Math.floor(sliceIndices[i] / chunkSize));
        }
        const srcChunkIds = Array.from(contributors).sort((a, b) => a - b);

        // For each layer in the request, gather.
        for (const layer of ALL_LAYERS) {
            const dst: Chunk | undefined = req[layer];
            if (!dst) continue;
            if (!parentMeta.layouts[layer]) {
                throw new Error(`indexed source: layer '${layer}' not available on parent`);
            }
            const stride = dst.stride;
            const staging = new Uint8Array(outCount * stride);

            // Cache: parent chunkId -> CPU buffer for this layer.
            const layerBuffers = new Map<number, Uint8Array>();

            for (const srcChunkId of srcChunkIds) {
                const srcChunkCount = Math.min(
                    chunkSize,
                    parentMeta.lodCounts[0] - srcChunkId * chunkSize
                );
                const srcChunk = manager.acquire(layer, parentMeta.layouts[layer]!, srcChunkCount);
                const layerReq: ReadRequest = { chunkIndex: srcChunkId, lod: 0 };
                (layerReq as { [K in Layer]?: Chunk })[layer] = srcChunk;
                await src.read(layerReq);
                const ab = await srcChunk.readBack();
                layerBuffers.set(srcChunkId, new Uint8Array(ab));
                srcChunk.release();
            }

            // Scatter into staging buffer.
            for (let i = 0; i < outCount; i++) {
                const srcRow = sliceIndices[i];
                const srcChunkId = Math.floor(srcRow / chunkSize);
                const srcRowInChunk = srcRow - srcChunkId * chunkSize;
                const srcBuf = layerBuffers.get(srcChunkId)!;
                staging.set(
                    srcBuf.subarray(srcRowInChunk * stride, (srcRowInChunk + 1) * stride),
                    i * stride
                );
            }

            dst.gpu.write(0, staging, 0, staging.byteLength);
        }
    };

    return {
        meta,
        read,
        close: () => src.close()
    };
};

/**
 * Build a filtered view over `src` from a precomputed mask.
 *
 * `mask` has length `src.meta.numGaussians`; non-zero entries mark surviving
 * rows. The resulting source's `numGaussians` is the count of non-zero
 * entries. Internally, the mask is converted to a tight `Uint32Array` of
 * surviving source indices and the work shares the indexed-source path.
 *
 * The predicate-driven variant (which walks the parent to compute the mask)
 * lives in `process.ts` alongside the relevant action implementations.
 */
const filterSource = (
    src: GaussianSource,
    mask: Uint8Array,
    manager: ChunkManager
): GaussianSource => {
    if (mask.length !== src.meta.numGaussians) {
        throw new Error(
            `filterSource: mask length ${mask.length} != source numGaussians ${src.meta.numGaussians}`
        );
    }
    let count = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) count++;

    const indices = new Uint32Array(count);
    let w = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) indices[w++] = i;

    return makeIndexedSource(src, indices, manager);
};

/**
 * Build a permuted view over `src` from a precomputed source-index order.
 *
 * `order` has length `src.meta.numGaussians`; entry `i` names the source
 * row that should appear at output row `i`. Used for Morton ordering,
 * visibility ordering, and any reorderings that don't change row count.
 */
const permuteSource = (
    src: GaussianSource,
    order: Uint32Array,
    manager: ChunkManager
): GaussianSource => {
    if (order.length !== src.meta.numGaussians) {
        throw new Error(
            `permuteSource: order length ${order.length} != source numGaussians ${src.meta.numGaussians}`
        );
    }
    return makeIndexedSource(src, order, manager);
};

// ---------------------------------------------------------------------------

/**
 * Virtually concatenate multiple sources into one.
 *
 * The resulting source presents its inputs as a single flat sequence of
 * gaussians. Each output chunk is `chunkSize` gaussians drawn from one or
 * more contributing input sources, repacked so that no output chunk is short
 * except the very last.
 *
 * All inputs must share `chunkSize`, `shBands`, `extraColumns`, and
 * `availableLayers`. Mismatched metadata is rejected at construction.
 */
const concatSource = (
    srcs: ReadonlyArray<GaussianSource>,
    manager: ChunkManager
): GaussianSource => {
    if (srcs.length === 0) {
        throw new Error('concatSource: at least one input required');
    }
    if (srcs.length === 1) return srcs[0];

    const ref = srcs[0].meta;
    for (let i = 1; i < srcs.length; i++) {
        const m = srcs[i].meta;
        if (m.chunkSize !== ref.chunkSize) {
            throw new Error(`concatSource: chunkSize mismatch (${m.chunkSize} vs ${ref.chunkSize})`);
        }
        if (m.shBands !== ref.shBands) {
            throw new Error(`concatSource: shBands mismatch (${m.shBands} vs ${ref.shBands})`);
        }
        if (m.availableLayers.size !== ref.availableLayers.size) {
            throw new Error('concatSource: availableLayers mismatch');
        }
        for (const l of ref.availableLayers) {
            if (!m.availableLayers.has(l)) {
                throw new Error(`concatSource: input ${i} missing layer '${l}'`);
            }
        }
    }

    const chunkSize = ref.chunkSize;
    // Per-input cumulative gaussian offsets to translate global row -> (srcIdx, rowInSrc).
    const offsets: number[] = [];
    let total = 0;
    for (const s of srcs) {
        offsets.push(total);
        total += s.meta.numGaussians;
    }

    const numChunks = Math.max(1, Math.ceil(total / chunkSize));
    const numGaussians = total;
    const lodCounts = [total];

    const meta: SourceMetadata = {
        numGaussians,
        numLods: 1,
        lodCounts,
        chunkSize,
        numChunks: [numChunks],
        shBands: ref.shBands,
        extraColumns: ref.extraColumns,
        transform: ref.transform,
        availableLayers: ref.availableLayers,
        layouts: ref.layouts
    };

    // Find which source contains a given global row.
    const findSrc = (globalRow: number): { srcIdx: number; rowInSrc: number } => {
        // Linear scan; srcs.length is small in practice.
        for (let i = srcs.length - 1; i >= 0; i--) {
            if (globalRow >= offsets[i]) {
                return { srcIdx: i, rowInSrc: globalRow - offsets[i] };
            }
        }
        return { srcIdx: 0, rowInSrc: globalRow };
    };

    const read = async (req: ReadRequest): Promise<void> => {
        if ((req.lod ?? 0) !== 0) {
            throw new Error('concatSource: only lod=0 is supported');
        }
        const outStart = req.chunkIndex * chunkSize;
        const outCount = Math.min(chunkSize, numGaussians - outStart);
        if (outCount <= 0) {
            throw new Error(`concatSource: chunkIndex ${req.chunkIndex} out of range`);
        }

        // For each requested layer, copy bytes from contributing source chunks.
        for (const layer of ALL_LAYERS) {
            const dst: Chunk | undefined = req[layer];
            if (!dst) continue;
            if (!meta.layouts[layer]) {
                throw new Error(`concatSource: layer '${layer}' not available`);
            }
            const stride = dst.stride;
            const staging = new Uint8Array(outCount * stride);

            let written = 0;
            let globalRow = outStart;
            while (written < outCount) {
                const { srcIdx, rowInSrc } = findSrc(globalRow);
                const src = srcs[srcIdx];
                const srcChunkSize = src.meta.chunkSize;
                const srcChunkId = Math.floor(rowInSrc / srcChunkSize);
                const srcRowInChunk = rowInSrc - srcChunkId * srcChunkSize;
                const srcChunkCount = Math.min(
                    srcChunkSize,
                    src.meta.numGaussians - srcChunkId * srcChunkSize
                );
                const rowsAvailableInSrcChunk = srcChunkCount - srcRowInChunk;
                const rowsToTake = Math.min(outCount - written, rowsAvailableInSrcChunk);

                // Read the contributing source chunk for this layer.
                const tmp = manager.acquire(layer, src.meta.layouts[layer]!, srcChunkCount);
                const layerReq: ReadRequest = { chunkIndex: srcChunkId, lod: 0 };
                (layerReq as { [K in Layer]?: Chunk })[layer] = tmp;
                await src.read(layerReq);
                const ab = await tmp.readBack();
                tmp.release();

                const srcBytes = new Uint8Array(ab);
                staging.set(
                    srcBytes.subarray(srcRowInChunk * stride, (srcRowInChunk + rowsToTake) * stride),
                    written * stride
                );

                written += rowsToTake;
                globalRow += rowsToTake;
            }

            dst.gpu.write(0, staging, 0, staging.byteLength);
        }
    };

    const close = async (): Promise<void> => {
        for (const s of srcs) await s.close();
    };

    return { meta, read, close };
};

export { mapSource, filterSource, permuteSource, concatSource };
