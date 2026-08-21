# Third-party notices

RouteRec's original code, configuration, tests, and documentation are licensed
under the Apache License 2.0. The files and components below retain their
upstream terms. This document is informational and does not replace those
licenses.

## MIT-licensed RecBole-derived baseline code

The following files contain code adapted from MIT-licensed RecBole-family
implementations:

- `src/routerec/models/duorec.py`
  ([RuihongQiu/DuoRec](https://github.com/RuihongQiu/DuoRec))
- `src/routerec/models/fearec.py`
  ([sudaada/FEARec](https://github.com/sudaada/FEARec))
- `src/routerec/models/fdsa.py`
  ([RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole))

The upstream repositories carry the following license notice:

```text
MIT License

Copyright (c) 2020 RUCAIBox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache-2.0 baseline code

`src/routerec/models/difsr.py` contains an adaptation of
[AIM-SE/DIF-SR](https://github.com/AIM-SE/DIF-SR), which is licensed under the
Apache License 2.0. The repository's top-level `LICENSE` contains the
applicable license text. The file has been modified for RouteRec's RecBole
1.2.x execution surface.

## Algorithmic references without vendored source

The project-specific implementations in `src/routerec/models/bsarec.py` and
`src/routerec/models/tisasrec.py` follow the respective papers and cite the
official repositories as algorithmic references. They are not vendored copies
of files from those repositories:

- [yehjin-shin/BSARec](https://github.com/yehjin-shin/BSARec)
- [JiachengLi1995/TiSASRec](https://github.com/JiachengLi1995/TiSASRec)

Those upstream repositories did not declare a repository license when this
notice was prepared on 2026-08-21. Do not copy upstream source files into
RouteRec without first obtaining compatible permission.
