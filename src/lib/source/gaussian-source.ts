import { type Transform } from '../utils';
import { type Chunk } from './chunk';
import { type LayerLayout } from './chunk-manager';
import { type ExtraColumn, type Layer, type SHBands } from './layout';

/**
 * Static description of a {@link GaussianSource}'s contents — what's in it and
 * how it's laid out. Populated at `open()` time; never changes thereafter.
 *
 * `chunkSize` is the gaussian count per chunk; all chunks are this size except
 * the final one in each LOD, which is `lodCounts[lod] % chunkSize` (or
 * `chunkSize` if the count divides evenly).
 *
 * `layouts` exposes the byte stride and named field map for each available
 * layer, used by callers when acquiring chunks from a {@link ChunkManager}.
 */
type SourceMetadata = {
    readonly numGaussians: number;
    readonly numLods: number;
    /** Gaussian counts per LOD. `lodCounts[0]` matches `numGaussians`. */
    readonly lodCounts: ReadonlyArray<number>;
    /** Gaussians per chunk (all chunks are this size except the last per LOD). */
    readonly chunkSize: number;
    /** Number of chunks per LOD: `Math.ceil(lodCounts[lod] / chunkSize)`. */
    readonly numChunks: ReadonlyArray<number>;
    /** SH band count present in the source. */
    readonly shBands: SHBands;
    /** Extra non-standard columns mapped to the `other` layer. */
    readonly extraColumns: ReadonlyArray<ExtraColumn>;
    /** Coordinate-space transform; applied lazily when consumed. */
    readonly transform: Transform;
    /** Which layers the source can serve. */
    readonly availableLayers: ReadonlySet<Layer>;
    /** Per-layer stride + field map. Keyed by layer; only present for available layers. */
    readonly layouts: Readonly<Partial<Record<Layer, LayerLayout>>>;
};

/**
 * A single read request to a {@link GaussianSource}.
 *
 * The caller passes destination chunks for whichever layers it wants filled
 * for the given `(chunkIndex, lod)`. Layers omitted from the request are
 * skipped. All passed chunks must have the same `count`, which must equal
 * `meta.chunkSize` for non-final chunks or the trailing count for the last.
 */
type ReadRequest = {
    readonly chunkIndex: number;
    readonly lod?: number;
    readonly position?: Chunk;
    readonly geometric?: Chunk;
    readonly color?: Chunk;
    readonly other?: Chunk;
};

/**
 * Lazy, chunked, GPU-backed view onto gaussian splat data.
 *
 * Sources are opened over a file (or derived from another source via a
 * combinator) and expose only metadata up front — no gaussian data is loaded
 * at open time except for formats whose decode is fundamentally whole-blob
 * (SPZ, MJS). Data is materialized into caller-allocated chunks on demand
 * via {@link GaussianSource.read}.
 *
 * Memory ownership is on the caller: chunks are acquired from a
 * `ChunkManager`, filled by `read`, used, and released back to the manager's
 * pool. The source itself never holds long-lived GPU memory on the caller's
 * behalf.
 */
interface GaussianSource {
    readonly meta: SourceMetadata;

    /**
     * Fill the caller's destination chunks with data for the given chunk
     * index. Layer fields present in the request are filled; absent layers
     * are skipped. All passed chunks must share the same `count` matching
     * the source's reported chunk size for the requested index.
     */
    read(request: ReadRequest): Promise<void>;

    /**
     * Release any open file handles or internal decode state.
     * Idempotent; safe to call multiple times.
     */
    close(): Promise<void>;
}

export { type GaussianSource, type ReadRequest, type SourceMetadata };
