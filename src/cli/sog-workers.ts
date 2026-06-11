import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';

import { Column, DataTable, type SogWorkers } from '../lib';

type PoolTask = {
    message: any;
    transfer: ArrayBuffer[];
    resolve: (result: any) => void;
    reject: (err: Error) => void;
};

// Pool of worker threads running the SOG writer's CPU-heavy steps
// (quantize1d and WebP encoding) off the main thread. Workers are spawned
// lazily on demand and run one task at a time; dist/sog-worker.mjs is built
// alongside the CLI bundle.
class SogWorkerPool {
    private threads: Worker[] = [];
    private idle: Worker[] = [];
    private queue: PoolTask[] = [];
    private active = new Map<Worker, PoolTask>();
    private maxThreads = Math.max(1, Math.min(6, availableParallelism() - 1));

    run(message: any, transfer: ArrayBuffer[]): Promise<any> {
        return new Promise((resolve, reject) => {
            this.queue.push({ message, transfer, resolve, reject });
            this.dispatch();
        });
    }

    async destroy() {
        await Promise.all(this.threads.map(thread => thread.terminate()));
        this.threads.length = 0;
        this.idle.length = 0;
    }

    private dispatch() {
        while (this.queue.length > 0) {
            const thread = this.idle.pop() ?? (this.threads.length < this.maxThreads ? this.spawn() : null);
            if (!thread) {
                return;
            }
            const task = this.queue.shift();
            this.active.set(thread, task);
            thread.postMessage(task.message, task.transfer);
        }
    }

    private spawn() {
        const thread = new Worker(new URL('sog-worker.mjs', import.meta.url));

        thread.on('message', (response: { result?: any, error?: string }) => {
            const task = this.active.get(thread);
            this.active.delete(thread);
            this.idle.push(thread);
            if (response.error) {
                task.reject(new Error(response.error));
            } else {
                task.resolve(response.result);
            }
            this.dispatch();
        });

        thread.on('error', (err: Error) => {
            const task = this.active.get(thread);
            this.active.delete(thread);
            this.threads.splice(this.threads.indexOf(thread), 1);
            task?.reject(err);
            this.dispatch();
        });

        this.threads.push(thread);
        return thread;
    }
}

/**
 * Creates worker-backed SogWorkers executors for parallel SOG writing.
 * Call destroy() once writing completes to terminate the threads.
 *
 * @returns The executors and a destroy function that terminates the pool.
 */
const createSogWorkers = (): { workers: SogWorkers, destroy: () => Promise<void> } => {
    const pool = new SogWorkerPool();

    return {
        workers: {
            quantize1d: async (dataTable: DataTable, k?: number, alpha?: number) => {
                // compact copies: column data may be views into larger buffers
                // and the originals must remain usable on the main thread
                const columns = dataTable.columns.map(c => ({ name: c.name, data: c.data.slice() }));
                const result = await pool.run(
                    { type: 'quantize1d', columns, k, alpha },
                    columns.map(c => c.data.buffer as ArrayBuffer)
                );
                return {
                    centroids: new DataTable([new Column('data', result.centroids)]),
                    labels: new DataTable(result.labels.map((c: { name: string, data: Uint8Array }) => new Column(c.name, c.data)))
                };
            },
            encodeWebp: (rgba: Uint8Array, width: number, height: number) => pool.run(
                { type: 'encodeWebp', rgba, width, height },
                [rgba.buffer as ArrayBuffer]
            )
        },
        destroy: () => pool.destroy()
    };
};

export { createSogWorkers };
