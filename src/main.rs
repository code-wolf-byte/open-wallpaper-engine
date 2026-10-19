//! The `wp-engine` binary: the desktop app when run without arguments,
//! otherwise the command line in [`cli`].

mod cli;

fn main() {
    cli::main();
}
