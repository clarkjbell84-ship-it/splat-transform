import { type Transform } from '../utils';
import { type Chunk } from './chunk';
import { type ChunkManager, type LayerLayout } from './chunk-manager';
import { type GaussianSource, type ReadRequest, type SourceMetadata } from './gaussian-source';
import {
    colorFields,
    colorStride,
    GEOMETRIC_STRIDE,
    geometricFields,
    type ExtraColumn,
    type Layer,
    otherLayout,
    positionFields,
    POSITION_STRIDE,
    type SHBands
} from './layout';

/**
 * Per-layer, per-LOD, per-chunk CPU-resident byte storage.
 *
 * Indexed as `buffers[layer][lod][chunkIndex]`. `undefined` for layers the
 * source doesn't carry.
 */
type LayerChunkBuffers = {
    [L in Layer]?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
};

/**
 * Lightweight CPU-resident source. Constructed directly from a set of
 * per-layer per-LOD per-chunk `ArrayBuffer`s. Used as:
 *
 * - The result of {@link compact} (densified output of a lazy combinator).
 * - A synthetic source for tests.
 * - The fallback materialization for whole-blob formats (SPZ, MJS) once the
 *   format's decoder has run.
 *
 * The source holds the provided buffers for its entire lifetime. `close()`
 * drops the references so they can be garbage collected.
 */
class InMemorySource implements GaussianSource {
    meta: SourceMetadata;
    private buffers: LayerChunkBuffers | null;

    constructor(meta: SourceMetadata, buffers: LayerChunkBuffers) {
        this.meta = meta;
        this.buffers = buffers;
    }

    async read(request: ReadRequest): Promise<void> {
        if (this.buffers === null) {
            throw new Error('InMemorySource.read: source has been closed');
        }
        const lod = request.lod ?? 0;
        const { chunkIndex } = request;

        const fill = (chunk: Chunk | undefined, layer: Layer): void => {
            if (!chunk) return;
            const layerBuffers = this.buffers![layer];
            if (!layerBuffers) {
                throw new Error(`InMemorySource: layer '${layer}' not available`);
            }
            const buf = layerBuffers[lod]?.[chunkIndex];
            if (buf === undefined) {
                throw new Error(
                    `InMemorySource: missing buffer for layer='${layer}' lod=${lod} chunkIndex=${chunkIndex}`
                );
            }
            if (buf.byteLength !== chunk.count * chunk.stride) {
                throw new Error(
                    `InMemorySource: buffer size mismatch for layer='${layer}' lod=${lod} chunk=${chunkIndex}: expected ${chunk.count * chunk.stride}, got ${buf.byteLength}`
                );
            }
            chunk.gpu.write(0, new Uint8Array(buf), 0, buf.byteLength);
        };

        fill(request.position, 'position');
        fill(request.geometric, 'geometric');
        fill(request.color, 'color');
        fill(request.other, 'other');
    }

    async close(): Promise<void> {
        this.buffers = null;
    }
}

/**
 * Convenience constructor for an `InMemorySource` built from raw layer buffers.
 *
 * The layouts and `numChunks` are derived from the provided buffers and the
 * supplied basic properties (SH band count, extra columns).
 */
const createInMemorySource = (params: {
    numGaussians: number;
    chunkSize: number;
    shBands: SHBands;
    extraColumns?: ReadonlyArray<ExtraColumn>;
    transform: Transform;
    /** Per-LOD per-chunk position-layer buffers, or `undefined` if the source lacks positions. */
    position?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
    geometric?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
    color?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
    other?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
}): InMemorySource => {
    const {
        numGaussians, chunkSize, shBands, transform,
        position, geometric, color, other
    } = params;
    const extras = params.extraColumns ?? [];

    const availableLayers = new Set<Layer>();
    const layouts: Partial<Record<Layer, LayerLayout>> = {};
    if (position) {
        availableLayers.add('position');
        layouts.position = { stride: POSITION_STRIDE, fields: positionFields() };
    }
    if (geometric) {
        availableLayers.add('geometric');
        layouts.geometric = { stride: GEOMETRIC_STRIDE, fields: geometricFields() };
    }
    if (color) {
        availableLayers.add('color');
        layouts.color = { stride: colorStride(shBands), fields: colorFields(shBands) };
    }
    if (other) {
        availableLayers.add('other');
        const ol = otherLayout(extras);
        layouts.other = { stride: ol.stride, fields: ol.fields };
    }

    // numChunks from whichever layer is present.
    const firstLayer = position ?? geometric ?? color ?? other;
    if (!firstLayer) {
        throw new Error('createInMemorySource: at least one layer must be provided');
    }
    const numLods = firstLayer.length;
    const lodCounts: number[] = [];
    const numChunks: number[] = [];
    for (let l = 0; l < numLods; l++) {
        const chunkCount = firstLayer[l].length;
        numChunks.push(chunkCount);
        // Total gaussians for this lod = numGaussians for lod 0, otherwise derive from
        // chunk count and chunk size (last chunk size is unknown without inspection).
        // For now record full chunkSize per chunk; the final chunk's effective count
        // is known by the consumer via meta.lodCounts. We rely on the caller passing
        // numGaussians for lod 0 only; lower LODs are expected to use the same chunkSize
        // and have their lodCounts derived by callers that know them. For step 1 we
        // assume single-LOD or that the caller supplies a more elaborate factory.
        lodCounts.push(l === 0 ? numGaussians : chunkCount * chunkSize);
    }

    const meta: SourceMetadata = {
        numGaussians,
        numLods,
        lodCounts,
        chunkSize,
        numChunks,
        shBands,
        extraColumns: extras,
        transform,
        availableLayers,
        layouts
    };

    const buffers: LayerChunkBuffers = {};
    if (position) buffers.position = position;
    if (geometric) buffers.geometric = geometric;
    if (color) buffers.color = color;
    if (other) buffers.other = other;

    return new InMemorySource(meta, buffers);
};

/**
 * Densify a source into a fresh {@link InMemorySource} by reading every chunk
 * of every available layer in order. Useful for:
 *
 * - Tests that need to compare two sources byte-for-byte.
 * - Materializing the output of lazy combinators (filter, permute, concat) so
 *   subsequent random-access reads don't re-decode the parent every time.
 *
 * Allocates one chunk per available layer at a time (via `manager`), reads
 * the parent into it, copies the GPU buffer back to a fresh `ArrayBuffer`,
 * and releases. Peak GPU memory = one chunk per available layer.
 */
const compact = async (
    src: GaussianSource,
    manager: ChunkManager
): Promise<InMemorySource> => {
    const { meta } = src;
    const layers: Layer[] = [];
    for (const l of ['position', 'geometric', 'color', 'other'] as Layer[]) {
        if (meta.availableLayers.has(l)) layers.push(l);
    }

    // Per-LOD per-chunk per-layer materialized buffers.
    const out: { [L in Layer]?: ArrayBuffer[][] } = {};
    for (const l of layers) {
        out[l] = [];
        for (let lod = 0; lod < meta.numLods; lod++) {
            out[l]![lod] = [];
        }
    }

    for (let lod = 0; lod < meta.numLods; lod++) {
        const total = meta.lodCounts[lod];
        for (let k = 0; k < meta.numChunks[lod]; k++) {
            const count = Math.min(meta.chunkSize, total - k * meta.chunkSize);
            const chunks: Partial<Record<Layer, Chunk>> = {};
            for (const l of layers) {
                chunks[l] = manager.acquire(l, meta.layouts[l]!, count);
            }
            await src.read({
                chunkIndex: k,
                lod,
                position: chunks.position,
                geometric: chunks.geometric,
                color: chunks.color,
                other: chunks.other
            });
            for (const l of layers) {
                const c = chunks[l]!;
                const ab = await c.readBack();
                out[l]![lod].push(ab);
            }
            for (const l of layers) chunks[l]!.release();
        }
    }

    return new InMemorySource(meta, out);
};

export { InMemorySource, createInMemorySource, compact };
