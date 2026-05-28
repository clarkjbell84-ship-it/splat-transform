// Layout & type constants
export {
    type Layer,
    type SHBands,
    type ChunkField,
    type ChunkFieldMap,
    type ExtraColumn,
    SH_REST_COUNTS,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout
} from './layout';

// Chunk + manager
export { type Chunk, CHUNK_BUFFER_USAGE } from './chunk';
export { type ChunkManager, type LayerLayout, createChunkManager } from './chunk-manager';

// Source contract
export { type GaussianSource, type ReadRequest, type SourceMetadata } from './gaussian-source';

// Combinators
export { mapSource, filterSource, permuteSource, concatSource } from './combinators';

// In-memory backing + compact
export { InMemorySource, createInMemorySource, compact } from './in-memory-source';

// DataTable -> GaussianSource compatibility adapter (used by readers during the 3.0 migration)
export { dataTableToSource } from './data-table-to-source';

// Caching wrapper
export { cached, type CachedGaussianSource, type CacheControls } from './cached';
