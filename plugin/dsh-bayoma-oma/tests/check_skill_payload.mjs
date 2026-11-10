import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const skillDir = path.join(root, 'skills', 'bayoma-oma');
const file = path.join(skillDir, 'SKILL.md');

const text = fs.readFileSync(file, 'utf8');
const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
if (!m) throw new Error('no frontmatter');
const meta = {};
for (const line of m[1].split(/\r?\n/)) {
  const i = line.indexOf(':');
  if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
}
const content = text.slice(text.indexOf('---', 3) + 3).replace(/^\r?\n/, '');

console.log('name        :', meta.name);
console.log('description :', (meta.description || '').slice(0, 40) + '...');
console.log('content len :', content.length);
console.log('has 异步一节   :', content.includes('## 6. 异步数据'));
console.log('has 6.1        :', content.includes('### 6.1'));
console.log('has section 11 :', content.includes('## 11. 使用说明'));
console.log('has 9.1         :', content.includes('### 9.1'));
console.log('has 11.1       :', content.includes('### 11.1'));
console.log('has 11.6       :', content.includes('### 11.6'));
console.log('has 10.1       :', content.includes('### 10.1'));
console.log('mentions condense:', content.includes('condense.py'));
console.log('mentions 未做  :', content.includes('本实现未做'));

const required = [
  ['## 6. 异步数据', true],
  ['### 6.1', true],
  ['## 11. 使用说明', true],
  ['### 9.1', true],
  ['### 10.1', true],
  ['### 11.1', true],
  ['### 11.6', true],
  ['本实现未做', false],
];
const missing = required.filter(([marker, want]) => content.includes(marker) !== want);
if (missing.length > 0) {
  throw new Error('技能正文缺少或多出标记：' + missing.map(([m]) => m).join(', '));
}

const registered = {
  name: meta.name,
  description: meta.description,
  content,
  resourceBase: { kind: 'directory', path: skillDir },
  source: 'runtime',
};
console.log('register payload name ok :', /^[a-z0-9]+(-[a-z0-9]+)*$/.test(registered.name));
