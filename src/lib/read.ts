import { DataTable } from './data-table';
import { ReadFileSystem, ZipReadFileSystem } from './io/read';
import { readKsplat, readLcc, readMjs, readPly, readSog, readSplat, readSpz } from './readers';
import { type GaussianSource, dataTableToSource } from './source';
import { Options, Param } from './types';

/**
 * Supported input file formats for Gaussian splat data.
 *
 * - `ply` - PLY format (standard 3DGS training output)
 * - `splat` - Antimatter15 splat format
 * - `ksplat` - Kevin Kwok's compressed splat format
 * - `spz` - Niantic Labs compressed format
 * - `sog` - PlayCanvas SOG format (WebP-compressed)
 * - `lcc` - XGrids LCC format
 * - `mjs` - JavaScript module generator
 */
type InputFormat = 'mjs' | 'ksplat' | 'splat' | 'sog' | 'ply' | 'spz' | 'lcc';

// Strip a trailing `?...` querystring and/or `#...` fragment from the
// *basename* so that extension sniffing works for URL-shaped inputs.
const stripQueryAndHash = (filename: string): string => {
    const lastSep = Math.max(filename.lastIndexOf('/'), filename.lastIndexOf('\\'));
    const basenameStart = lastSep + 1;
    const q = filename.slice(basenameStart).search(/[?#]/);
    return q < 0 ? filename : filename.slice(0, basenameStart + q);
};

/**
 * Determines the input format based on file extension.
 */
const getInputFormat = (filename: string): InputFormat => {
    const lowerFilename = stripQueryAndHash(filename).toLowerCase();

    if (lowerFilename.endsWith('.mjs')) {
        return 'mjs';
    } else if (lowerFilename.endsWith('.ksplat')) {
        return 'ksplat';
    } else if (lowerFilename.endsWith('.splat')) {
        return 'splat';
    } else if (lowerFilename.endsWith('.sog') || lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.spz')) {
        return 'spz';
    } else if (lowerFilename.endsWith('.lcc')) {
        return 'lcc';
    }

    throw new Error(`Unsupported input file type: ${filename}`);
};

/**
 * Options for reading a Gaussian splat file.
 */
type ReadFileOptions = {
    /** Path to the input file. */
    filename: string;
    /** The format of the input file. */
    inputFormat: InputFormat;
    /** Processing options. */
    options: Options;
    /** Parameters for generator modules (.mjs files). */
    params: Param[];
    /** File system abstraction for reading files. */
    fileSystem: ReadFileSystem;
};

/** Dispatch to the appropriate format-specific reader; returns one or more DataTables. */
const readDataTables = async (readFileOptions: ReadFileOptions): Promise<DataTable[]> => {
    const { filename, inputFormat, options, params, fileSystem } = readFileOptions;

    if (inputFormat === 'mjs') {
        return [await readMjs(filename, params)];
    }
    if (inputFormat === 'sog') {
        const lowerFilename = stripQueryAndHash(filename).toLowerCase();
        if (lowerFilename.endsWith('.sog')) {
            const source = await fileSystem.createSource(filename);
            const zipFs = new ZipReadFileSystem(source);
            try {
                return [await readSog(zipFs, 'meta.json')];
            } finally {
                zipFs.close();
            }
        }
        return [await readSog(fileSystem, filename)];
    }
    if (inputFormat === 'lcc') {
        return await readLcc(fileSystem, filename, options);
    }

    const source = await fileSystem.createSource(filename);
    try {
        if (inputFormat === 'ply') return [await readPly(source)];
        if (inputFormat === 'ksplat') return [await readKsplat(source)];
        if (inputFormat === 'splat') return [await readSplat(source)];
        if (inputFormat === 'spz') return [await readSpz(source)];
        throw new Error(`Unsupported input format: ${inputFormat}`);
    } finally {
        source.close();
    }
};

/**
 * Reads a Gaussian splat file and returns its data as one or more DataTables.
 *
 * Supports multiple input formats including PLY, splat, ksplat, spz, SOG, and LCC.
 * Some formats (like LCC) may return multiple DataTables for different LOD levels.
 *
 * @param readFileOptions - Options specifying the file to read and how to read it.
 * @returns Promise resolving to an array of DataTables containing the splat data.
 */
const readFile = async (readFileOptions: ReadFileOptions): Promise<DataTable[]> => {
    return readDataTables(readFileOptions);
};

/**
 * Reads a Gaussian splat file and returns its data as one or more `GaussianSource`s.
 *
 * This is the new 3.0 entry point for the chunked source API. Each reader's
 * `DataTable` output is wrapped via `dataTableToSource` — the canonical
 * layered chunk shape, ready for the new `processSource` / new writers.
 *
 * Once individual readers are migrated to emit `GaussianSource` natively
 * (skipping the intermediate `DataTable`), this dispatcher will route to
 * them directly without the conversion step.
 */
const readFileAsSource = async (readFileOptions: ReadFileOptions): Promise<GaussianSource[]> => {
    const tables = await readDataTables(readFileOptions);
    return tables.map(t => dataTableToSource(t));
};

export {
    readFile,
    readFileAsSource,
    getInputFormat,
    type InputFormat,
    type ReadFileOptions
};
