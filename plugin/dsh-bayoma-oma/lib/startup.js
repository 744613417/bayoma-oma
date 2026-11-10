// dsh-bayoma-oma — register the textbook-based BAYOMA skill.
//
// The skill body lives in skills/bayoma-oma/SKILL.md. At activation this module
// reads that file, validates its frontmatter, and registers it as a runtime
// skill so the model can load it by name. Resource files beside SKILL.md stay
// addressable through resourceBase.
import { readFileSync, existsSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SKILL_DIR = join(ROOT, 'skills', 'bayoma-oma');
const NAME_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

/** Parse the leading YAML frontmatter into a flat key/value map. */
function frontmatter(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
  if (!m) return null;
  const out = {};
  for (const line of m[1].split(/\r?\n/)) {
    if (!line || /^\s/.test(line) || line.trim().startsWith('#')) continue;
    const i = line.indexOf(':');
    if (i <= 0) continue;
    const k = line.slice(0, i).trim();
    if (out[k] !== undefined) continue;
    let v = line.slice(i + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
      v = v.slice(1, -1);
    }
    out[k] = v;
  }
  return out;
}

export default function apply(ctx) {
  if (!ctx?.skills) {
    console.warn('[dsh-bayoma-oma] ctx.skills unavailable');
    return;
  }
  const file = join(SKILL_DIR, 'SKILL.md');
  if (!existsSync(file) || !statSync(file).isFile()) {
    console.warn('[dsh-bayoma-oma] SKILL.md missing');
    return;
  }
  const text = readFileSync(file, 'utf8');
  const fm = frontmatter(text);
  if (!fm || !NAME_RE.test(String(fm.name || '')) || !fm.description) {
    console.warn('[dsh-bayoma-oma] invalid frontmatter');
    return;
  }
  const content = text.slice(text.indexOf('---', 3) + 3).replace(/^\r?\n/, '');
  ctx.skills.register({
    name: fm.name,
    description: fm.description,
    content,
    resourceBase: { kind: 'directory', path: SKILL_DIR },
    source: 'runtime',
  });
  console.log('[dsh-bayoma-oma] registered bayoma-oma');
}
