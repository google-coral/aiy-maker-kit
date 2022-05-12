# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process the PyCoral docs from Sphinx to optimize them for Coral website."""

import argparse
import os
import re

from bs4 import BeautifulSoup


def relocate_h2id(soup):
    """Moves the anchor ID to the H2 tag, from the wrapper DIV."""
    for h2 in soup.find_all('h2'):
        div = h2.find_parent('div')
        if div.has_attr('id') and not h2.has_attr('id'):
            # print('Move ID: ' + div['id'])
            h2['id'] = div['id']
            del div['id']
        # Also delete embedded <a> tag
        if h2.find('a'):
            h2.find('a').extract()
    return soup


def strip_li_paragraphs(soup):
    """Removes paragraph tags from inside list items."""
    for p in soup.select('li>p:first-child'):
        p.unwrap()
    return soup


def demote_h1s(soup):
    """Demotes H1 tags to H3."""
    for h in soup.find_all('h1'):
        h.name = 'h3'
    return soup


def process(file):
    """Runs all the cleanup functions."""
    print('Post-processing ' + file)
    soup = BeautifulSoup(open(file), 'html.parser')
    soup = relocate_h2id(soup)
    soup = demote_h1s(soup)
    soup = strip_li_paragraphs(soup)
    with open(file, 'w') as output:
        output.write(str(soup))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--file', required=True, help='File path of HTML file(s).')
    args = parser.parse_args()

    # Accept a directory or single file
    if os.path.isdir(args.file):
        for file in os.listdir(args.file):
            if os.path.splitext(file)[1] == '.md':
                process(os.path.join(args.file, file))
    else:
        process(args.file)


if __name__ == '__main__':
    main()
