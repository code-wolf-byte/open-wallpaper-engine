#!/usr/bin/env bash
# Build a wp-engine release archive in dist/, optionally bumping the version
# and tagging it.
#
#   scripts/release.sh                     package the current version
#   scripts/release.sh --version 0.2.0     bump Cargo.toml/Cargo.lock first
#   scripts/release.sh --version 0.2.0 --tag
#                                          ...then commit the bump and tag v0.2.0
#   scripts/release.sh --web               include web wallpapers (Linux; bundles CEF)
#   scripts/release.sh --skip-checks       skip fmt/clippy/tests
#
# Produces dist/wp-engine-<version>-<os>-<arch>[-web].tar.gz and a .sha256
# next to it. Nothing is pushed: the script prints the push command.
set -euo pipefail
cd "$(dirname "$0")/.."

version="" web=0 tag=0 checks=1
while (($#)); do
    case $1 in
    --version) version=${2:?--version needs a value}; shift 2 ;;
    --web) web=1; shift ;;
    --tag) tag=1; shift ;;
    --skip-checks) checks=0; shift ;;
    -h | --help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
    *) echo "release: unknown option '$1' (see --help)" >&2; exit 2 ;;
    esac
done

die() { echo "release: $*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
((web)) && [[ $os != linux ]] && die "--web is Linux-only (the CEF bundle is laid out for Linux)"

if ((tag)); then
    [[ -n $version ]] || die "--tag needs --version"
    [[ -z $(git status --porcelain) ]] || die "the working tree has uncommitted changes"
    git rev-parse -q --verify "refs/tags/v$version" >/dev/null && die "tag v$version already exists"
fi

if [[ -n $version ]]; then
    [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || die "version must look like 1.2.3 or 1.2.3-rc.1"
    step "Setting version $version"
    # The first line-initial `version = ` is [package]'s; dependency versions
    # are inline tables. perl rather than sed -i: same syntax on GNU and BSD.
    perl -0pi -e "s/^version = \"[^\"]*\"/version = \"$version\"/m" Cargo.toml
    cargo update --offline -q -p wp-engine # refresh Cargo.lock's own entry
fi
version=$(cargo pkgid | sed 's/.*[#@]//')

if ((checks)); then
    step "Checking"
    cargo fmt --check
    cargo clippy --locked --release -q
    cargo test --locked --release -q
fi

step "Building wp-engine $version for $os-$arch$( ((web)) && echo ' with web support')"
features=()
((web)) && features=(--features web)
cargo build --locked --release "${features[@]}"

name="wp-engine-$version-$os-$arch"
((web)) && name+="-web"
dist="dist/$name"
step "Packaging $dist"
rm -rf "$dist" "$dist.tar.gz" "$dist.tar.gz.sha256"
mkdir -p "$dist"
cp target/release/wp-engine LICENSE README.md "$dist/"
cp -R extras "$dist/"
if ((web)); then
    # What cef-dll-sys puts next to the binary and Chromium loads at runtime;
    # build.rs gives the binary an $ORIGIN rpath to find libcef.so here.
    for f in libcef.so libEGL.so libGLESv2.so libvk_swiftshader.so libvulkan.so.1 \
        vk_swiftshader_icd.json chrome-sandbox icudtl.dat v8_context_snapshot.bin \
        chrome_100_percent.pak chrome_200_percent.pak resources.pak locales; do
        [[ -e target/release/$f ]] || die "target/release/$f is missing — did the web build finish?"
        cp -R "target/release/$f" "$dist/"
    done
fi
# Debug symbols are most of the size (libcef.so ships with ~1.2 GB of them).
strip "$dist/wp-engine" 2>/dev/null || true
for lib in "$dist"/*.so "$dist"/*.so.*; do
    [[ -e $lib ]] && { strip --strip-unneeded "$lib" 2>/dev/null || true; }
done

tar -C dist -czf "$dist.tar.gz" "$name"
if command -v sha256sum >/dev/null; then
    (cd dist && sha256sum "$name.tar.gz" >"$name.tar.gz.sha256")
else
    (cd dist && shasum -a 256 "$name.tar.gz" >"$name.tar.gz.sha256")
fi
rm -rf "$dist"

if ((tag)); then
    step "Tagging v$version"
    git add Cargo.toml Cargo.lock
    git diff --cached --quiet || git commit -q -m "release: v$version"
    git tag -a "v$version" -m "wp-engine $version"
fi

step "Done"
ls -lh "$dist.tar.gz"
cat "$dist.tar.gz.sha256"
((tag)) && echo "Push with: git push origin $(git branch --show-current) v$version"
exit 0
