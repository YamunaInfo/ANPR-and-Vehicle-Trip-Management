// Re-export the full ANPR pipeline from the canonical ai/ module.
// The pipeline source lives in ai/src/anpr-pipeline.ts.
// This stub keeps existing frontend imports (`./lib/anpr-pipeline`) working.
export * from '@workspace/ai';
