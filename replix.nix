{ pkgs }: {
  deps = [
    pkgs.python39Full
    pkgs.ffmpeg
    pkgs.gcc
    pkgs.pkgconfig
    pkgs.zlib
  ];
}
