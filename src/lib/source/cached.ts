import { type Chunk } from './chunk';
import { type ChunkManager } from './chunk-manager';
import { type GaussianSource, type ReadRequest, type SourceMetadata } from './gaussian-source';
import { type Layer } from './layout';

/**
 * Cache key for a single (layer, chunkIndex, lod) entry.
 */
type CacheKey = string;

const keyOf = (layer: Layer, chunkIndex: number, lod: number): CacheKey => `${layer}/${lod}/${chunkIndex}`;

/**
 * Per-(layer, chunkIndex, lod) cached entry.
 *
 * `bytes` holds the decoded CPU mirror of the chunk's GPU contents. Pinned
 * entries are never evicted while their pin count is > 0.
 */
type CacheEntry = {
    layer: Layer;
    chunkIndex: number;
    lod: number;
    bytes: ArrayBuffer;
    pinCount: number;
};

const ALL_LAYERS: ReadonlyArray<Layer> = ['position', 'geometric', 'color', 'other'];

/**
 * The extra pin/unpin/prefetch API a cached source exposes on top of the
 * base `GaussianSource` contract.
 */
interface CacheControls {
    /** Pin the (layer, chunkIndex, lod) entry to prevent LRU eviction. */
    pin(layer: Layer, chunkIndex: number, lod?: number): void;
    /** Decrement the pin count on an entry; once 0 it becomes evictable. */
    unpin(layer: Layer, chunkIndex: number, lod?: number): void;
    /**
     * Hint that the listed (layer, chunkIndex, lod) entries will be needed
     * soon. The cache may pre-fetch them in the background.
     */
    prefetch(reqs: ReadonlyArray<{ layer: Layer; chunkIndex: number; lod?: number }>): void;
    /** Bytes currently held by the cache (sum of entry buffers). */
    readonly cacheBytes: number;
}

type CachedGaussianSource = GaussianSource & CacheControls;

/**
 * Wrap a source with an LRU cache of decoded chunk bytes.
 *
 * On `read`, for each requested layer:
 *  - cache hit (entry exists) -> upload cached bytes directly to the caller's chunk
 *  - cache miss              -> delegate to parent.read into a temp chunk,
 *                                readBack the bytes, cache them, then upload
 *                                to the caller's chunk
 *
 * Pinned entries are kept regardless of budget. Unpinned entries are evicted
 * in least-recently-used order whenever `cacheBytes > budgetBytes`.
 *
 * Caching is CPU-side: entries hold `ArrayBuffer`s, not GPU buffers. This
 * avoids doubling GPU memory pressure but pays a per-hit CPU->GPU upload.
 * For the common multi-pass use case (LOD partition, decimation iterations)
 * the savings on disk decode dwarf the upload cost.
 */
const cached = (
    source: GaussianSource,
    manager: ChunkManager,
    options?: { budgetBytes?: number }
): CachedGaussianSource => {
    const budgetBytes = options?.budgetBytes ?? 1 * 1024 * 1024 * 1024;

    // LRU: Map preserves insertion order; we delete-then-set on access to move to end.
    const entries = new Map<CacheKey, CacheEntry>();
    let cacheBytes = 0;

    const evictIfNeeded = (): void => {
        if (cacheBytes <= budgetBytes) return;
        // Walk in insertion (LRU) order; skip pinned.
        for (const [k, entry] of entries) {
            if (cacheBytes <= budgetBytes) return;
            if (entry.pinCount > 0) continue;
            entries.delete(k);
            cacheBytes -= entry.bytes.byteLength;
        }
    };

    const touch = (key: CacheKey, entry: CacheEntry): void => {
        entries.delete(key);
        entries.set(key, entry);
    };

    const fetch = async (layer: Layer, chunkIndex: number, lod: number): Promise<CacheEntry> => {
        const key = keyOf(layer, chunkIndex, lod);
        const existing = entries.get(key);
        if (existing) {
            touch(key, existing);
            return existing;
        }
        const layout = source.meta.layouts[layer];
        if (!layout) {
            throw new Error(`cached: layer '${layer}' not available on parent`);
        }
        const count = Math.min(
            source.meta.chunkSize,
            source.meta.lodCounts[lod] - chunkIndex * source.meta.chunkSize
        );
        const tmp = manager.acquire(layer, layout, count);
        const layerReq: ReadRequest = { chunkIndex, lod };
        (layerReq as { [K in Layer]?: Chunk })[layer] = tmp;
        await source.read(layerReq);
        const bytes = await tmp.readBack();
        tmp.release();

        const entry: CacheEntry = { layer, chunkIndex, lod, bytes, pinCount: 0 };
        entries.set(key, entry);
        cacheBytes += bytes.byteLength;
        evictIfNeeded();
        return entry;
    };

    const meta: SourceMetadata = source.meta;

    const read = async (request: ReadRequest): Promise<void> => {
        const lod = request.lod ?? 0;
        const { chunkIndex } = request;
        for (const layer of ALL_LAYERS) {
            const dst: Chunk | undefined = request[layer];
            if (!dst) continue;
            const entry = await fetch(layer, chunkIndex, lod);
            dst.gpu.write(0, new Uint8Array(entry.bytes), 0, entry.bytes.byteLength);
        }
    };

    const close = async (): Promise<void> => {
        entries.clear();
        cacheBytes = 0;
        await source.close();
    };

    const pin = (layer: Layer, chunkIndex: number, lod = 0): void => {
        const key = keyOf(layer, chunkIndex, lod);
        const entry = entries.get(key);
        if (!entry) {
            // Pre-pin: create a placeholder entry so the next fetch will populate it
            // and respect the pin. Simpler: just call fetch lazily on actual read.
            // Here we record the pin against a future entry by storing a zero-byte
            // sentinel — but since pin must be matched by unpin, and the caller's
            // expectation is "ensure this stays loaded", the cleanest behavior is
            // to ignore pins on absent entries. Document: pin only takes effect
            // once the entry has been read at least once.
            return;
        }
        entry.pinCount++;
    };

    const unpin = (layer: Layer, chunkIndex: number, lod = 0): void => {
        const key = keyOf(layer, chunkIndex, lod);
        const entry = entries.get(key);
        if (!entry) return;
        if (entry.pinCount > 0) entry.pinCount--;
    };

    const prefetch = (
        reqs: ReadonlyArray<{ layer: Layer; chunkIndex: number; lod?: number }>
    ): void => {
        // Fire-and-forget: kicks off fetches but doesn't await. Errors are
        // swallowed; if the caller subsequently does a real read of the same
        // key, it'll re-attempt and surface any error then.
        for (const r of reqs) {
            const lod = r.lod ?? 0;
            const key = keyOf(r.layer, r.chunkIndex, lod);
            if (entries.has(key)) continue;
            void fetch(r.layer, r.chunkIndex, lod).catch((): undefined => undefined);
        }
    };

    return {
        meta,
        read,
        close,
        pin,
        unpin,
        prefetch,
        get cacheBytes() { return cacheBytes; }
    };
};

export { cached, type CachedGaussianSource, type CacheControls };
