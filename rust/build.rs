use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

fn main() {
    let source_root = Path::new("src");
    let mut source_hashes = BTreeMap::new();
    collect_source_hashes(source_root, source_root, &mut source_hashes);

    let generated_source = render_source_hashes(&source_hashes);
    let output_dir = std::env::var("OUT_DIR").unwrap();
    let output_path = Path::new(&output_dir).join("native_source_hashes.rs");
    fs::write(output_path, generated_source).unwrap();

    println!("cargo:rerun-if-changed=src");
}

fn collect_source_hashes(
    source_root: &Path,
    directory: &Path,
    source_hashes: &mut BTreeMap<String, String>,
) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();

        if path.is_dir() {
            collect_source_hashes(source_root, &path, source_hashes);
            continue;
        }

        if !path.extension().is_some_and(|extension| extension == "rs") {
            continue;
        }

        let relative_path = path
            .strip_prefix(source_root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        println!("cargo:rerun-if-changed={}", path.display());

        let contents = fs::read(&path).unwrap();
        let digest = format!("{:x}", Sha256::digest(contents));
        source_hashes.insert(relative_path, digest);
    }
}

fn render_source_hashes(source_hashes: &BTreeMap<String, String>) -> String {
    let entries = source_hashes
        .iter()
        .map(|(path, digest)| format!("({path:?},{digest:?})"))
        .collect::<Vec<_>>()
        .join(",");

    format!("&[{entries}]")
}
