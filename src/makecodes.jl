const vertex_codes = [
    "\x15\x45\x51\x54\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA",
    "\x15\x45\x51\x55\x56\x59\x5A\x65\x66\x6A\x95\x96\x9A\xA6\xAA",
    "\x01\x05\x11\x15\x41\x45\x51\x55\x56\x5A\x66\x6A\x96\x9A\xA6\xAA",
    "\x01\x15\x16\x45\x46\x51\x52\x55\x56\x5A\x66\x6A\x96\x9A\xA6\xAA\xAB",
    "\x15\x45\x54\x55\x56\x59\x5A\x65\x69\x6A\x95\x99\x9A\xA9\xAA",
    "\x05\x15\x45\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xAA",
    "\x05\x15\x45\x55\x56\x59\x5A\x66\x6A\x96\x9A\xAA",
    "\x05\x15\x16\x45\x46\x55\x56\x59\x5A\x66\x6A\x96\x9A\xAA\xAB",
    "\x04\x05\x14\x15\x44\x45\x54\x55\x59\x5A\x69\x6A\x99\x9A\xA9\xAA",
    "\x05\x15\x45\x55\x56\x59\x5A\x69\x6A\x99\x9A\xAA",
    "\x05\x15\x45\x55\x56\x59\x5A\x6A\x9A\xAA",
    "\x05\x15\x16\x45\x46\x55\x56\x59\x5A\x5B\x6A\x9A\xAA\xAB",
    "\x04\x15\x19\x45\x49\x54\x55\x58\x59\x5A\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x05\x15\x19\x45\x49\x55\x56\x59\x5A\x69\x6A\x99\x9A\xAA\xAE",
    "\x05\x15\x19\x45\x49\x55\x56\x59\x5A\x5E\x6A\x9A\xAA\xAE",
    "\x05\x15\x1A\x45\x4A\x55\x56\x59\x5A\x5B\x5E\x6A\x9A\xAA\xAB\xAE\xAF",
    "\x15\x51\x54\x55\x56\x59\x65\x66\x69\x6A\x95\xA5\xA6\xA9\xAA",
    "\x11\x15\x51\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\xA5\xA6\xAA",
    "\x11\x15\x51\x55\x56\x5A\x65\x66\x6A\x96\xA6\xAA",
    "\x11\x15\x16\x51\x52\x55\x56\x5A\x65\x66\x6A\x96\xA6\xAA\xAB",
    "\x14\x15\x54\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x99\xA5\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x9A\xA6\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x96\x9A\xA6\xAA\xAB",
    "\x15\x16\x55\x56\x5A\x66\x6A\x6B\x96\x9A\xA6\xAA\xAB",
    "\x14\x15\x54\x55\x59\x5A\x65\x69\x6A\x99\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x9A\xAA",
    "\x15\x16\x55\x56\x59\x5A\x66\x6A\x6B\x9A\xAA\xAB",
    "\x14\x15\x19\x54\x55\x58\x59\x5A\x65\x69\x6A\x99\xA9\xAA\xAE",
    "\x15\x19\x55\x59\x5A\x69\x6A\x6E\x99\x9A\xA9\xAA\xAE",
    "\x15\x19\x55\x56\x59\x5A\x69\x6A\x6E\x9A\xAA\xAE",
    "\x15\x1A\x55\x56\x59\x5A\x6A\x6B\x6E\x9A\xAA\xAB\xAE\xAF",
    "\x10\x11\x14\x15\x50\x51\x54\x55\x65\x66\x69\x6A\xA5\xA6\xA9\xAA",
    "\x11\x15\x51\x55\x56\x65\x66\x69\x6A\xA5\xA6\xAA",
    "\x11\x15\x51\x55\x56\x65\x66\x6A\xA6\xAA",
    "\x11\x15\x16\x51\x52\x55\x56\x65\x66\x67\x6A\xA6\xAA\xAB",
    "\x14\x15\x54\x55\x59\x65\x66\x69\x6A\xA5\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\xA6\xAA",
    "\x15\x16\x55\x56\x5A\x65\x66\x6A\x6B\xA6\xAA\xAB",
    "\x14\x15\x54\x55\x59\x65\x69\x6A\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\xAA",
    "\x15\x16\x55\x56\x59\x5A\x65\x66\x69\x6A\x6B\xAA\xAB",
    "\x14\x15\x19\x54\x55\x58\x59\x65\x69\x6A\x6D\xA9\xAA\xAE",
    "\x15\x19\x55\x59\x5A\x65\x69\x6A\x6E\xA9\xAA\xAE",
    "\x15\x19\x55\x56\x59\x5A\x65\x66\x69\x6A\x6E\xAA\xAE",
    "\x15\x55\x56\x59\x5A\x66\x69\x6A\x6B\x6E\x9A\xAA\xAB\xAE\xAF",
    "\x10\x15\x25\x51\x54\x55\x61\x64\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x11\x15\x25\x51\x55\x56\x61\x65\x66\x69\x6A\xA5\xA6\xAA\xBA",
    "\x11\x15\x25\x51\x55\x56\x61\x65\x66\x6A\x76\xA6\xAA\xBA",
    "\x11\x15\x26\x51\x55\x56\x62\x65\x66\x67\x6A\x76\xA6\xAA\xAB\xBA\xBB",
    "\x14\x15\x25\x54\x55\x59\x64\x65\x66\x69\x6A\xA5\xA9\xAA\xBA",
    "\x15\x25\x55\x65\x66\x69\x6A\x7A\xA5\xA6\xA9\xAA\xBA",
    "\x15\x25\x55\x56\x65\x66\x69\x6A\x7A\xA6\xAA\xBA",
    "\x15\x26\x55\x56\x65\x66\x6A\x6B\x7A\xA6\xAA\xAB\xBA\xBB",
    "\x14\x15\x25\x54\x55\x59\x64\x65\x69\x6A\x79\xA9\xAA\xBA",
    "\x15\x25\x55\x59\x65\x66\x69\x6A\x7A\xA9\xAA\xBA",
    "\x15\x25\x55\x56\x59\x5A\x65\x66\x69\x6A\x7A\xAA\xBA",
    "\x15\x55\x56\x5A\x65\x66\x69\x6A\x6B\x7A\xA6\xAA\xAB\xBA\xBB",
    "\x14\x15\x29\x54\x55\x59\x65\x68\x69\x6A\x6D\x79\xA9\xAA\xAE\xBA\xBE",
    "\x15\x29\x55\x59\x65\x69\x6A\x6E\x7A\xA9\xAA\xAE\xBA\xBE",
    "\x15\x55\x59\x5A\x65\x66\x69\x6A\x6E\x7A\xA9\xAA\xAE\xBA\xBE",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x6B\x6E\x7A\xAA\xAB\xAE\xBA\xBF",
    "\x45\x51\x54\x55\x56\x59\x65\x95\x96\x99\x9A\xA5\xA6\xA9\xAA",
    "\x41\x45\x51\x55\x56\x59\x5A\x65\x66\x95\x96\x99\x9A\xA5\xA6\xAA",
    "\x41\x45\x51\x55\x56\x5A\x66\x95\x96\x9A\xA6\xAA",
    "\x41\x45\x46\x51\x52\x55\x56\x5A\x66\x95\x96\x9A\xA6\xAA\xAB",
    "\x44\x45\x54\x55\x56\x59\x5A\x65\x69\x95\x96\x99\x9A\xA5\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x65\x6A\x95\x96\x99\x9A\xA6\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x66\x6A\x95\x96\x99\x9A\xA6\xAA\xAB",
    "\x45\x46\x55\x56\x5A\x66\x6A\x96\x9A\x9B\xA6\xAA\xAB",
    "\x44\x45\x54\x55\x59\x5A\x69\x95\x99\x9A\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x69\x6A\x95\x96\x99\x9A\xA9\xAA\xAE",
    "\x45\x55\x56\x59\x5A\x6A\x95\x96\x99\x9A\xAA",
    "\x45\x46\x55\x56\x59\x5A\x6A\x96\x9A\x9B\xAA\xAB",
    "\x44\x45\x49\x54\x55\x58\x59\x5A\x69\x95\x99\x9A\xA9\xAA\xAE",
    "\x45\x49\x55\x59\x5A\x69\x6A\x99\x9A\x9E\xA9\xAA\xAE",
    "\x45\x49\x55\x56\x59\x5A\x6A\x99\x9A\x9E\xAA\xAE",
    "\x45\x4A\x55\x56\x59\x5A\x6A\x9A\x9B\x9E\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x56\x59\x65\x66\x69\x95\x96\x99\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x59\x65\x66\x6A\x95\x96\x9A\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x5A\x65\x66\x6A\x95\x96\x9A\xA5\xA6\xAA\xAB",
    "\x51\x52\x55\x56\x5A\x66\x6A\x96\x9A\xA6\xA7\xAA\xAB",
    "\x54\x55\x56\x59\x65\x69\x6A\x95\x99\x9A\xA5\xA6\xA9\xAA",
    "\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA",
    "\x15\x45\x51\x55\x56\x59\x5A\x65\x66\x6A\x95\x96\x9A\xA6\xAA\xAB",
    "\x55\x56\x5A\x66\x6A\x96\x9A\xA6\xAA\xAB",
    "\x54\x55\x59\x5A\x65\x69\x6A\x95\x99\x9A\xA5\xA9\xAA\xAE",
    "\x15\x45\x54\x55\x56\x59\x5A\x65\x69\x6A\x95\x99\x9A\xA9\xAA\xAE",
    "\x15\x45\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xA6\xA9\xAA\xAB\xAE",
    "\x55\x56\x59\x5A\x66\x6A\x96\x9A\xA6\xAA\xAB",
    "\x54\x55\x58\x59\x5A\x69\x6A\x99\x9A\xA9\xAA\xAD\xAE",
    "\x55\x59\x5A\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x55\x56\x59\x5A\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x55\x56\x59\x5A\x6A\x9A\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x65\x66\x69\x95\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x65\x66\x69\x6A\x95\x96\xA5\xA6\xA9\xAA\xBA",
    "\x51\x55\x56\x65\x66\x6A\x95\x96\xA5\xA6\xAA",
    "\x51\x52\x55\x56\x65\x66\x6A\x96\xA6\xA7\xAA\xAB",
    "\x54\x55\x59\x65\x66\x69\x6A\x95\x99\xA5\xA6\xA9\xAA\xBA",
    "\x15\x51\x54\x55\x56\x59\x65\x66\x69\x6A\x95\xA5\xA6\xA9\xAA\xBA",
    "\x15\x51\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x9A\xA5\xA6\xA9\xAA\xAB\xBA",
    "\x55\x56\x5A\x65\x66\x6A\x96\x9A\xA6\xAA\xAB",
    "\x54\x55\x59\x65\x69\x6A\x95\x99\xA5\xA9\xAA",
    "\x15\x54\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x99\x9A\xA5\xA6\xA9\xAA\xAE\xBA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x9A\xA6\xA9\xAA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x96\x9A\xA6\xAA\xAB",
    "\x54\x55\x58\x59\x65\x69\x6A\x99\xA9\xAA\xAD\xAE",
    "\x55\x59\x5A\x65\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\x99\x9A\xA9\xAA\xAE",
    "\x15\x55\x56\x59\x5A\x66\x69\x6A\x9A\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x61\x64\x65\x66\x69\x95\xA5\xA6\xA9\xAA\xBA",
    "\x51\x55\x61\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xB6\xBA",
    "\x51\x55\x56\x61\x65\x66\x6A\xA5\xA6\xAA\xB6\xBA",
    "\x51\x55\x56\x62\x65\x66\x6A\xA6\xA7\xAA\xAB\xB6\xBA\xBB",
    "\x54\x55\x64\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xB9\xBA",
    "\x55\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x55\x56\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x55\x56\x65\x66\x6A\xA6\xAA\xAB\xBA\xBB",
    "\x54\x55\x59\x64\x65\x69\x6A\xA5\xA9\xAA\xB9\xBA",
    "\x55\x59\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x15\x55\x56\x59\x5A\x65\x66\x69\x6A\xA5\xA6\xA9\xAA\xBA",
    "\x15\x55\x56\x5A\x65\x66\x69\x6A\xA6\xAA\xAB\xBA\xBB",
    "\x54\x55\x59\x65\x68\x69\x6A\xA9\xAA\xAD\xAE\xB9\xBA\xBE",
    "\x55\x59\x65\x69\x6A\xA9\xAA\xAE\xBA\xBE",
    "\x15\x55\x59\x5A\x65\x66\x69\x6A\xA9\xAA\xAE\xBA\xBE",
    "\x55\x56\x59\x5A\x65\x66\x69\x6A\xAA\xAB\xAE\xBA\xBF",
    "\x40\x41\x44\x45\x50\x51\x54\x55\x95\x96\x99\x9A\xA5\xA6\xA9\xAA",
    "\x41\x45\x51\x55\x56\x95\x96\x99\x9A\xA5\xA6\xAA",
    "\x41\x45\x51\x55\x56\x95\x96\x9A\xA6\xAA",
    "\x41\x45\x46\x51\x52\x55\x56\x95\x96\x97\x9A\xA6\xAA\xAB",
    "\x44\x45\x54\x55\x59\x95\x96\x99\x9A\xA5\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\xA6\xAA",
    "\x45\x46\x55\x56\x5A\x95\x96\x9A\x9B\xA6\xAA\xAB",
    "\x44\x45\x54\x55\x59\x95\x99\x9A\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\xAA",
    "\x45\x46\x55\x56\x59\x5A\x95\x96\x99\x9A\x9B\xAA\xAB",
    "\x44\x45\x49\x54\x55\x58\x59\x95\x99\x9A\x9D\xA9\xAA\xAE",
    "\x45\x49\x55\x59\x5A\x95\x99\x9A\x9E\xA9\xAA\xAE",
    "\x45\x49\x55\x56\x59\x5A\x95\x96\x99\x9A\x9E\xAA\xAE",
    "\x45\x55\x56\x59\x5A\x6A\x96\x99\x9A\x9B\x9E\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x65\x95\x96\x99\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x65\x66\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x51\x55\x56\x65\x66\x95\x96\x9A\xA5\xA6\xAA",
    "\x51\x52\x55\x56\x66\x95\x96\x9A\xA6\xA7\xAA\xAB",
    "\x54\x55\x59\x65\x69\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x45\x51\x54\x55\x56\x59\x65\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x45\x51\x55\x56\x59\x5A\x65\x66\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xAB\xEA",
    "\x55\x56\x5A\x66\x6A\x95\x96\x9A\xA6\xAA\xAB",
    "\x54\x55\x59\x65\x69\x95\x99\x9A\xA5\xA9\xAA",
    "\x45\x54\x55\x56\x59\x5A\x65\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xAE\xEA",
    "\x45\x55\x56\x59\x5A\x6A\x95\x96\x99\x9A\xA6\xA9\xAA",
    "\x45\x55\x56\x59\x5A\x66\x6A\x95\x96\x99\x9A\xA6\xAA\xAB",
    "\x54\x55\x58\x59\x69\x95\x99\x9A\xA9\xAA\xAD\xAE",
    "\x55\x59\x5A\x69\x6A\x95\x99\x9A\xA9\xAA\xAE",
    "\x45\x55\x56\x59\x5A\x69\x6A\x95\x96\x99\x9A\xA9\xAA\xAE",
    "\x45\x55\x56\x59\x5A\x6A\x96\x99\x9A\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x65\x95\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x65\x66\x95\x96\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x65\x66\x95\x96\xA5\xA6\xAA",
    "\x51\x52\x55\x56\x65\x66\x95\x96\xA5\xA6\xA7\xAA\xAB",
    "\x54\x55\x59\x65\x69\x95\x99\xA5\xA6\xA9\xAA",
    "\x51\x54\x55\x56\x59\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xBA\xEA",
    "\x51\x55\x56\x65\x66\x6A\x95\x96\x9A\xA5\xA6\xA9\xAA",
    "\x51\x55\x56\x5A\x65\x66\x6A\x95\x96\x9A\xA5\xA6\xAA\xAB",
    "\x54\x55\x59\x65\x69\x95\x99\xA5\xA9\xAA",
    "\x54\x55\x59\x65\x69\x6A\x95\x99\x9A\xA5\xA6\xA9\xAA",
    "\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA",
    "\x55\x56\x59\x5A\x65\x66\x6A\x95\x96\x9A\xA6\xA9\xAA\xAB",
    "\x54\x55\x58\x59\x65\x69\x95\x99\xA5\xA9\xAA\xAD\xAE",
    "\x54\x55\x59\x5A\x65\x69\x6A\x95\x99\x9A\xA5\xA9\xAA\xAE",
    "\x55\x56\x59\x5A\x65\x69\x6A\x95\x99\x9A\xA6\xA9\xAA\xAE",
    "\x55\x56\x59\x5A\x66\x69\x6A\x96\x99\x9A\xA6\xA9\xAA\xAB\xAE\xAF",
    "\x50\x51\x54\x55\x61\x64\x65\x95\xA5\xA6\xA9\xAA\xB5\xBA",
    "\x51\x55\x61\x65\x66\x95\xA5\xA6\xA9\xAA\xB6\xBA",
    "\x51\x55\x56\x61\x65\x66\x95\x96\xA5\xA6\xAA\xB6\xBA",
    "\x51\x55\x56\x65\x66\x6A\x96\xA5\xA6\xA7\xAA\xAB\xB6\xBA\xBB",
    "\x54\x55\x64\x65\x69\x95\xA5\xA6\xA9\xAA\xB9\xBA",
    "\x55\x65\x66\x69\x6A\x95\xA5\xA6\xA9\xAA\xBA",
    "\x51\x55\x56\x65\x66\x69\x6A\x95\x96\xA5\xA6\xA9\xAA\xBA",
    "\x51\x55\x56\x65\x66\x6A\x96\xA5\xA6\xAA\xAB\xBA\xBB",
    "\x54\x55\x59\x64\x65\x69\x95\x99\xA5\xA9\xAA\xB9\xBA",
    "\x54\x55\x59\x65\x66\x69\x6A\x95\x99\xA5\xA6\xA9\xAA\xBA",
    "\x55\x56\x59\x65\x66\x69\x6A\x95\x9A\xA5\xA6\xA9\xAA\xBA",
    "\x55\x56\x5A\x65\x66\x69\x6A\x96\x9A\xA5\xA6\xA9\xAA\xAB\xBA\xBB",
    "\x54\x55\x59\x65\x69\x6A\x99\xA5\xA9\xAA\xAD\xAE\xB9\xBA\xBE",
    "\x54\x55\x59\x65\x69\x6A\x99\xA5\xA9\xAA\xAE\xBA\xBE",
    "\x55\x59\x5A\x65\x66\x69\x6A\x99\x9A\xA5\xA6\xA9\xAA\xAE\xBA\xBE",
    "\x55\x56\x59\x5A\x65\x66\x69\x6A\x9A\xA6\xA9\xAA\xAB\xAE\xBA",
    "\x40\x45\x51\x54\x55\x85\x91\x94\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x41\x45\x51\x55\x56\x85\x91\x95\x96\x99\x9A\xA5\xA6\xAA\xEA",
    "\x41\x45\x51\x55\x56\x85\x91\x95\x96\x9A\xA6\xAA\xD6\xEA",
    "\x41\x45\x51\x55\x56\x86\x92\x95\x96\x97\x9A\xA6\xAA\xAB\xD6\xEA\xEB",
    "\x44\x45\x54\x55\x59\x85\x94\x95\x96\x99\x9A\xA5\xA9\xAA\xEA",
    "\x45\x55\x85\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xDA\xEA",
    "\x45\x55\x56\x85\x95\x96\x99\x9A\xA6\xAA\xDA\xEA",
    "\x45\x55\x56\x86\x95\x96\x9A\x9B\xA6\xAA\xAB\xDA\xEA\xEB",
    "\x44\x45\x54\x55\x59\x85\x94\x95\x99\x9A\xA9\xAA\xD9\xEA",
    "\x45\x55\x59\x85\x95\x96\x99\x9A\xA9\xAA\xDA\xEA",
    "\x45\x55\x56\x59\x5A\x85\x95\x96\x99\x9A\xAA\xDA\xEA",
    "\x45\x55\x56\x5A\x95\x96\x99\x9A\x9B\xA6\xAA\xAB\xDA\xEA\xEB",
    "\x44\x45\x54\x55\x59\x89\x95\x98\x99\x9A\x9D\xA9\xAA\xAE\xD9\xEA\xEE",
    "\x45\x55\x59\x89\x95\x99\x9A\x9E\xA9\xAA\xAE\xDA\xEA\xEE",
    "\x45\x55\x59\x5A\x95\x96\x99\x9A\x9E\xA9\xAA\xAE\xDA\xEA\xEE",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\x9B\x9E\xAA\xAB\xAE\xDA\xEA\xEF",
    "\x50\x51\x54\x55\x65\x91\x94\x95\x96\x99\xA5\xA6\xA9\xAA\xEA",
    "\x51\x55\x91\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xE6\xEA",
    "\x51\x55\x56\x91\x95\x96\x9A\xA5\xA6\xAA\xE6\xEA",
    "\x51\x55\x56\x92\x95\x96\x9A\xA6\xA7\xAA\xAB\xE6\xEA\xEB",
    "\x54\x55\x94\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xE9\xEA",
    "\x55\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x55\x56\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x55\x56\x95\x96\x9A\xA6\xAA\xAB\xEA\xEB",
    "\x54\x55\x59\x94\x95\x99\x9A\xA5\xA9\xAA\xE9\xEA",
    "\x55\x59\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x45\x55\x56\x59\x5A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x45\x55\x56\x5A\x95\x96\x99\x9A\xA6\xAA\xAB\xEA\xEB",
    "\x54\x55\x59\x95\x98\x99\x9A\xA9\xAA\xAD\xAE\xE9\xEA\xEE",
    "\x55\x59\x95\x99\x9A\xA9\xAA\xAE\xEA\xEE",
    "\x45\x55\x59\x5A\x95\x96\x99\x9A\xA9\xAA\xAE\xEA\xEE",
    "\x55\x56\x59\x5A\x95\x96\x99\x9A\xAA\xAB\xAE\xEA\xEF",
    "\x50\x51\x54\x55\x65\x91\x94\x95\xA5\xA6\xA9\xAA\xE5\xEA",
    "\x51\x55\x65\x91\x95\x96\xA5\xA6\xA9\xAA\xE6\xEA",
    "\x51\x55\x56\x65\x66\x91\x95\x96\xA5\xA6\xAA\xE6\xEA",
    "\x51\x55\x56\x66\x95\x96\x9A\xA5\xA6\xA7\xAA\xAB\xE6\xEA\xEB",
    "\x54\x55\x65\x94\x95\x99\xA5\xA6\xA9\xAA\xE9\xEA",
    "\x55\x65\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x51\x55\x56\x65\x66\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x51\x55\x56\x66\x95\x96\x9A\xA5\xA6\xAA\xAB\xEA\xEB",
    "\x54\x55\x59\x65\x69\x94\x95\x99\xA5\xA9\xAA\xE9\xEA",
    "\x54\x55\x59\x65\x69\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x55\x56\x59\x65\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xEA",
    "\x55\x56\x5A\x66\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xAB\xEA\xEB",
    "\x54\x55\x59\x69\x95\x99\x9A\xA5\xA9\xAA\xAD\xAE\xE9\xEA\xEE",
    "\x54\x55\x59\x69\x95\x99\x9A\xA5\xA9\xAA\xAE\xEA\xEE",
    "\x55\x59\x5A\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xAE\xEA\xEE",
    "\x55\x56\x59\x5A\x6A\x95\x96\x99\x9A\xA6\xA9\xAA\xAB\xAE\xEA",
    "\x50\x51\x54\x55\x65\x95\xA1\xA4\xA5\xA6\xA9\xAA\xB5\xBA\xE5\xEA\xFA",
    "\x51\x55\x65\x95\xA1\xA5\xA6\xA9\xAA\xB6\xBA\xE6\xEA\xFA",
    "\x51\x55\x65\x66\x95\x96\xA5\xA6\xA9\xAA\xB6\xBA\xE6\xEA\xFA",
    "\x51\x55\x56\x65\x66\x95\x96\xA5\xA6\xA7\xAA\xAB\xB6\xBA\xE6\xEA\xFB",
    "\x54\x55\x65\x95\xA4\xA5\xA6\xA9\xAA\xB9\xBA\xE9\xEA\xFA",
    "\x55\x65\x95\xA5\xA6\xA9\xAA\xBA\xEA\xFA",
    "\x51\x55\x65\x66\x95\x96\xA5\xA6\xA9\xAA\xBA\xEA\xFA",
    "\x55\x56\x65\x66\x95\x96\xA5\xA6\xAA\xAB\xBA\xEA\xFB",
    "\x54\x55\x65\x69\x95\x99\xA5\xA6\xA9\xAA\xB9\xBA\xE9\xEA\xFA",
    "\x54\x55\x65\x69\x95\x99\xA5\xA6\xA9\xAA\xBA\xEA\xFA",
    "\x55\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xBA\xEA\xFA",
    "\x55\x56\x65\x66\x6A\x95\x96\x9A\xA5\xA6\xA9\xAA\xAB\xBA\xEA",
    "\x54\x55\x59\x65\x69\x95\x99\xA5\xA9\xAA\xAD\xAE\xB9\xBA\xE9\xEA\xFE",
    "\x55\x59\x65\x69\x95\x99\xA5\xA9\xAA\xAE\xBA\xEA\xFE",
    "\x55\x59\x65\x69\x6A\x95\x99\x9A\xA5\xA6\xA9\xAA\xAE\xBA\xEA",
    "\x55\x56\x59\x5A\x65\x66\x69\x6A\x95\x96\x99\x9A\xA5\xA6\xA9\xAA\xAB\xAE\xBA\xEA",
]

# Lookup Tables & Gradients

struct Vertex4D
    dx::Float32
    dy::Float32
    dz::Float32
    dw::Float32
    xsvp::Int64
    ysvp::Int64
    zsvp::Int64
    wsvp::Int64

    function Vertex4D(xsv::Int, ysv::Int, zsv::Int, wsv::Int)
        ssv = Float32((xsv + ysv + zsv + wsv) * UNSKEW_4D)
        new(-xsv - ssv, -ysv - ssv, -zsv - ssv, -wsv - ssv,
            (xsv * PRIME_X)%Int64,
            (ysv * PRIME_Y)%Int64,
            (zsv * PRIME_Z)%Int64,
            (wsv * PRIME_W)%Int64)
    end
end
Vertex4D(v) = Vertex4D(((v >> 0) & 3) - 1, ((v >> 2) & 3) - 1,
                       ((v >> 4) & 3) - 1, ((v >> 6) & 3) - 1)

const offsets_4D = Vector{UInt16}(undef, 256)

function create_lattice_map()
    num_vertices = 0
    tab = falses(256)
    for i = 1:256
        codes = vertex_codes[i]
        num_vertices += sizeof(codes)
        offsets_4D[i] = num_vertices
        for v in codeunits(codes)
            tab[v+1] = true
        end
    end
    siz = count(tab)
    inv_map = zeros(UInt8, 256)
    vertices = Vector{UInt8}(undef, siz)
    vertex_map = Vector{Vertex4D}(undef, siz)
    r = findfirst(tab)
    i = 1
    while r !== nothing
        inv_map[r] = i
        vertices[i] = r-1
        vertex_map[i] = Vertex4D(r-1)
        i += 1
        r = findnext(tab, r+1)
    end
    io = IOBuffer()
    println(io, "const vertex_ind = [")
    for codes in vertex_codes
        print(io, "    ")
        for v in codeunits(codes)
            i = inv_map[v+1]
            print(io, i<10 ? " " : "", i, ',')
        end
        println(io)
    end
    println(io, "]")
    String(take!(io)), vertex_map, vertices, inv_map
end
