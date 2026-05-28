// Data table
export { Column, DataTable, combine, convertToSpace, computeSummary, sortMortonOrder, sortByVisibility, simplifyGaussians, getSHBands } from './data-table';
export type { TypedArray, ColumnType, Row, ColumnStats, SummaryData } from './data-table';

// Utils
export {
    fmtBytes, fmtCount, fmtDistance, fmtTime,
    logger, TextRenderer, Transform, WebPCodec
} from './utils';
export type { Bar, Group, LogEvent, Logger, MessageKind, Renderer, TextRendererOptions, Verbosity } from './utils';

// High-level read/write
export { readFile, readFileAsSource, getInputFormat } from './read';
export type { InputFormat, ReadFileOptions } from './read';

// New chunk-based source API (3.0)
export {
    type Chunk,
    type ChunkFieldMap,
    type ChunkManager,
    type GaussianSource,
    type Layer,
    type LayerLayout,
    type ReadRequest,
    type SHBands,
    type SourceMetadata,
    createChunkManager,
    createInMemorySource,
    InMemorySource,
    dataTableToSource,
    cached,
    mapSource,
    filterSource,
    permuteSource,
    concatSource,
    compact
} from './source';
export { writeFile, getOutputFormat } from './write';
export type { OutputFormat, WriteOptions } from './write';

// Processing
export { processDataTable } from './process';
export type {
    ProcessAction,
    ProcessOptions,
    Translate,
    Rotate,
    Scale,
    FilterNaN,
    FilterByValue,
    FilterBands,
    FilterBox,
    FilterSphere,
    FilterFloaters,
    FilterCluster,
    Param as ProcessParam,
    Lod,
    Summary,
    MortonOrder,
    Decimate
} from './process';

// File system abstractions
export { ReadStream, BufferedReadStream, MemoryReadFileSystem, UrlReadFileSystem, ZipReadFileSystem } from './io/read';
export type { ReadSource, ReadFileSystem, ProgressCallback, ZipEntry } from './io/read';
export { MemoryFileSystem, ZipFileSystem } from './io/write';
export type { FileSystem, Writer } from './io/write';

// Individual readers (for advanced use)
export { readKsplat, readLcc, readMjs, readPly, readSog, readSplat, readSpz } from './readers';

// Individual writers (for advanced use)
export { writeSog, writeSpz, writePly, writeCompressedPly, writeCsv, writeHtml, writeImage, writeLod, writeGlb, writeVoxel } from './writers';
export type { WriteImageOptions, WriteVoxelOptions, VoxelMetadata } from './writers';

// Renderer (for advanced use)
export { renderSplats, buildCameraBasis } from './render';
export type { Projection, RenderCamera, CameraBasis } from './render';

// Voxel
export { carve, fillExterior, fillFloor, filterCluster, filterFloaters, findClusterVoxelFlood, voxelizeToBuffer } from './voxel';
export type { NavSeed, NavSimplifyResult } from './voxel';

// Types
export type { CollisionMeshShape, Options, Param, DeviceCreator } from './types';

// Version
export { version, revision } from './version';
