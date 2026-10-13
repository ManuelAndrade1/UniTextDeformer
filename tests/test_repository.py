"""Check that documentation, configuration and figures are portable."""
import json
from pathlib import Path
import re
import struct
import unittest
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_documentation_links_and_images_are_included(self):
        for path in ROOT.rglob('*.md'):
            content = path.read_text()
            targets = re.findall(r'\]\(([^)]+)\)', content) + re.findall(r'\bsrc="([^"]+)"', content)
            for target in targets:
                if target.startswith(('https://', '#')):
                    continue
                destination = (path.parent / target.split('#')[0]).resolve()
                self.assertTrue(destination.is_file(), f'{path.name}: missing {target}')
                self.assertTrue(destination.is_relative_to(ROOT))

    def test_dependency_sources_are_public_https_urls(self):
        assets = json.loads((ROOT / 'configs/assets.json').read_text())
        allowed = {'github.com', 'huggingface.co', 'openaipublic.azureedge.net'}
        for entry in assets['repositories'] + assets['assets']:
            url = urlparse(entry['url'])
            self.assertEqual(url.scheme, 'https')
            self.assertIn(url.hostname, allowed)
            self.assertIsNone(url.username)
            self.assertFalse(Path(entry['path']).is_absolute())
            self.assertNotIn('..', Path(entry['path']).parts)

    def test_readme_results_agree_with_reference_values(self):
        summary = json.loads((ROOT / 'assets/tables/results.json').read_text())
        self.assertEqual((summary['case_count'], summary['steps'], summary['seed']), (20, 2500, 42))
        readme = (ROOT / 'README.md').read_text()
        latex = (ROOT / 'assets/tables/results.tex').read_text()
        for row in summary['results']:
            self.assertIn(f'{row["r_precision_percent"]:.1f}'.replace('.', ',') + ' %', readme)
            self.assertIn(f'{row["intersections_percent"]:.2f}'.replace('.', ',') + ' %', readme)
            self.assertIn(f'{row["r_precision_percent"]:.1f}' + r'\%', latex)
            self.assertIn(f'{row["intersections_percent"]:.2f}' + r'\%', latex)

    def test_figures_contain_no_embedded_source_metadata(self):
        for path in (ROOT / 'assets/images').glob('*.png'):
            data = path.read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            offset = 8
            while offset < len(data):
                length = struct.unpack('>I', data[offset:offset + 4])[0]
                kind = data[offset + 4:offset + 8]
                self.assertNotIn(kind, {b'tEXt', b'zTXt', b'iTXt', b'eXIf'})
                offset += length + 12


if __name__ == '__main__':
    unittest.main()
