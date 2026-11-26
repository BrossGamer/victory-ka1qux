//! Where a title's own stores of descriptor records are
//! (`VICTORY_PORT_DESCRIPTORS=1` with `VICTORY_DESCRIPTORS_PROFILE`).
//!
//! A title that fills the descriptor tables of its draws itself, one 32-byte
//! record at a time, can tell the layer which slots it wrote
//! (`victory-loader`'s `descriptor_sites` patches each store to mark its slot,
//! `victory-gpu`'s `port_descriptors` takes the marks). Nothing about a title
//! is built into this program: where the stores are is told by a profile the
//! user keeps, in the format of the title profile of `VICTORY_FSR`
//! (`crate::title_profile`: `key = value` lines, `#` starting a comment, byte
//! patterns of hex bytes with `??` for any byte), by byte patterns that must
//! each match exactly once in the executable sections of the mapped image: a
//! pattern that matches nowhere or twice (another build of the title) leaves
//! that store alone and says so (a store the patches miss is found by the
//! fallback of `port_descriptors`, so the heap is right either way, only
//! slower). The keys:
//!
//! ```text
//! # Only for this build of the executable (BLAKE3 of the file, as `run` prints it).
//! build = <64 hex digits>
//! # One line for each store: `<offset> <length> : <pattern>`, the store's
//! # offset from the start of the match and its length in bytes. The store is
//! # a `vmovups [base+index], ymm0` (VEX, no displacement); the bytes of the
//! # store must be fixed bytes of the pattern.
//! store = 10 5 : 81 E1 ?? ?? ... (hex bytes, `??` for any byte)
//! ```
//!
//! The idea, and the shape of the patch, are avatarmanz/VIctory-7900XTX's
//! ("Renderer port 2", GPL-3.0), which found these stores by address for one
//! build; the profile is this project's own, so that no code of a title is
//! kept in its source.

use anyhow::{Context, Result, bail, ensure};

use crate::title_profile::{matches, parse_pattern};

/// A section that holds code.
const EXECUTE: u32 = 0x2000_0000;

/// A parsed profile.
#[derive(Debug, Default, PartialEq)]
pub struct Profile {
    pub build: Option<String>,
    stores: Vec<StoreSite>,
}

#[derive(Debug, PartialEq)]
struct StoreSite {
    pattern: Vec<Option<u8>>,
    /// The store's offset in the match and its length.
    at: usize,
    len: usize,
}

/// A store of the title's: its address and length (for the loader).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Found {
    pub at: usize,
    pub len: usize,
}

impl Profile {
    pub fn parse(text: &str) -> Result<Self> {
        let mut build = None;
        let mut stores = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let at = || format!("profile line {}", n + 1);
            let (key, value) = line
                .split_once('=')
                .with_context(|| format!("{}: expected key = value", at()))?;
            let value = value.trim();
            match key.trim() {
                "build" => {
                    ensure!(
                        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()),
                        "{}: build is 64 hex digits",
                        at()
                    );
                    build = Some(value.to_ascii_lowercase());
                }
                "store" => stores.push(parse_store(value).with_context(at)?),
                other => bail!("{}: unknown key {other}", at()),
            }
        }
        ensure!(!stores.is_empty(), "a profile needs at least one store");
        Ok(Self { build, stores })
    }

    /// How many stores the profile names.
    pub fn len(&self) -> usize {
        self.stores.len()
    }
}

/// `<offset> <length> : <pattern>`.
fn parse_store(value: &str) -> Result<StoreSite> {
    let (place, pattern) = value
        .split_once(':')
        .context("store needs `<offset> <length> : <pattern>`")?;
    let mut numbers = place.split_whitespace().map(|n| {
        n.parse::<usize>()
            .with_context(|| format!("not a number: {n}"))
    });
    let at = numbers.next().context("store needs an offset")??;
    let len = numbers.next().context("store needs a length")??;
    ensure!(
        numbers.next().is_none(),
        "store takes an offset and a length, then : and the pattern"
    );
    let pattern = parse_pattern(pattern)?;
    ensure!(
        (5..=16).contains(&len),
        "the store is {len} bytes: a jump takes 5, an instruction at most 15"
    );
    ensure!(
        pattern
            .get(at..at + len)
            .is_some_and(|store| store.iter().all(Option::is_some)),
        "the store (offset {at}, {len} bytes) must lie on fixed bytes of the pattern"
    );
    Ok(StoreSite { pattern, at, len })
}

/// Each store of the profile found in `sections` (the executable sections:
/// the address of each and its bytes): the one place its pattern matches, or
/// why not.
pub fn locate(sections: &[(usize, &[u8])], profile: &Profile) -> Vec<Result<Found, String>> {
    profile
        .stores
        .iter()
        .map(|site| {
            let mut hits = sections.iter().flat_map(|&(address, code)| {
                matches(code, &site.pattern)
                    .into_iter()
                    .map(move |m| address + m)
            });
            match (hits.next(), hits.next()) {
                (Some(at), None) => Ok(Found {
                    at: at + site.at,
                    len: site.len,
                }),
                (None, _) => Err("its code is not in this build of the title".into()),
                (Some(_), Some(_)) => Err(format!(
                    "its pattern matches {} places in this build",
                    2 + hits.count()
                )),
            }
        })
        .collect()
}

/// The executable sections of a mapped image, as `locate` takes them.
///
/// # Safety
/// The image of `size` bytes is mapped at `base`, its code readable and
/// staying so as long as the slices are used.
unsafe fn code_of<'a>(
    base: usize,
    size: usize,
    sections: &[victory_pe::SectionInfo],
) -> Vec<(usize, &'a [u8])> {
    sections
        .iter()
        .filter(|s| s.characteristics & EXECUTE != 0)
        .filter_map(|s| {
            let len = match s.virtual_size {
                0 => s.file_size,
                vs => vs.min(s.file_size),
            } as usize;
            let at = base.checked_add(s.rva as usize)?;
            (s.rva as usize)
                .checked_add(len)
                .filter(|&end| end <= size)?;
            // SAFETY: by the caller; the section lies inside the image.
            Some((at, unsafe {
                std::slice::from_raw_parts(at as *const u8, len)
            }))
        })
        .collect()
}

/// What `patch_image` did.
#[derive(Debug, PartialEq)]
pub struct Patched {
    /// Stores patched to mark their slots.
    pub done: usize,
    /// What became of each of the others, for the log.
    pub notes: Vec<String>,
}

/// Finds the profile's stores in the image of `size` bytes mapped at `base`
/// (`sections` from its headers) and makes each one found exactly once tell
/// the cells at `marks` its slot (`victory_loader::descriptor_sites`).
///
/// # Safety
/// The image is mapped at `base`, whole to `size`, and none of its code runs
/// yet; `marks` stays valid, as the cells say, for as long as it runs.
pub unsafe fn patch_image(
    base: usize,
    size: usize,
    sections: &[victory_pe::SectionInfo],
    profile: &Profile,
    marks: usize,
) -> Patched {
    use victory_loader::descriptor_sites::{Hooked, Store, apply};
    // SAFETY: by the caller.
    let code = unsafe { code_of(base, size, sections) };
    let mut stores = Vec::new();
    let mut notes = Vec::new();
    // The profile's stores found, and the lines of those not.
    let mut lines = Vec::new();
    for (n, found) in locate(&code, profile).into_iter().enumerate() {
        match found {
            Ok(f) => {
                lines.push(n);
                stores.push(Store {
                    at: f.at,
                    len: f.len,
                });
            }
            Err(why) => notes.push(format!("store {}: not patched, {why}", n + 1)),
        }
    }
    // SAFETY: by the caller; the stores are inside the image's code.
    let done = unsafe { apply(marks, &stores) };
    for (n, hooked) in lines.into_iter().zip(&done) {
        if let Hooked::Failed(why) = hooked {
            notes.push(format!("store {}: not patched, {why}", n + 1));
        }
    }
    Patched {
        done: done.iter().filter(|d| **d == Hooked::Done).count(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic code: no title's bytes anywhere here.
    const STORE_A: &str = "A1 A2 ?? ?? A3 A4 10 20 30 40 50 B1 B2 ?? ?? C1";
    const STORE_B: &str = "D1 ?? D2 D3 ?? ?? 61 62 63 64 65 66 E1 E2 ?? E3 E4";
    const PROFILE: &str = "# a made-up profile\n\
        store = 6 5 : A1 A2 ?? ?? A3 A4 10 20 30 40 50 B1 B2 ?? ?? C1  # first\n\
        store = 6 6 : D1 ?? D2 D3 ?? ?? 61 62 63 64 65 66 E1 E2 ?? E3 E4\n";

    fn instance(pattern: &str, fill: u8) -> Vec<u8> {
        parse_pattern(pattern)
            .unwrap()
            .into_iter()
            .map(|p| p.unwrap_or(fill))
            .collect()
    }

    #[test]
    fn a_profile_parses_and_names_what_is_wrong_with_a_bad_one() {
        let profile = Profile::parse(PROFILE).unwrap();
        assert_eq!(profile.build, None);
        assert_eq!(profile.len(), 2);
        assert_eq!((profile.stores[0].at, profile.stores[0].len), (6, 5));
        assert_eq!(profile.stores[1].pattern.len(), 17);
        let with_build = format!("build = {}\n{PROFILE}", "AB".repeat(32));
        assert_eq!(
            Profile::parse(&with_build).unwrap().build,
            Some("ab".repeat(32))
        );
        for (bad, why) in [
            ("", "at least one store"),
            ("colour = 1", "unknown key"),
            ("store", "expected key = value"),
            ("build = 12\nstore = 0 5 : 01 02 03 04 05", "64 hex digits"),
            (
                "store = 0 5 01 02 03 04 05",
                "needs `<offset> <length> : <pattern>`",
            ),
            ("store = 0 : 01 02 03 04 05", "needs a length"),
            ("store = x 5 : 01 02 03 04 05", "not a number"),
            (
                "store = 0 5 1 : 01 02 03 04 05",
                "takes an offset and a length",
            ),
            ("store = 0 4 : 01 02 03 04 05", "a jump takes 5"),
            (
                "store = 0 17 : 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F 10 11",
                "a jump takes 5",
            ),
            ("store = 0 5 : 01 02 03 04", "fixed bytes of the pattern"),
            ("store = 1 5 : 01 02 03 04 05", "fixed bytes of the pattern"),
            ("store = 0 5 : 01 02 ?? 04 05", "fixed bytes of the pattern"),
            ("store = 0 5 : ?? ?? ?? ?? ??", "at least one fixed byte"),
            ("store = 0 5 : 01 02 03 04 zz", "not a hex byte"),
        ] {
            let error = format!("{:#}", Profile::parse(bad).unwrap_err());
            assert!(error.contains(why), "{bad:?}: {error}");
        }
        // Open bytes outside the store are fine.
        assert!(Profile::parse("store = 2 5 : ?? ?? 01 02 03 04 05 ??").is_ok());
    }

    /// Each pattern found once gives its store's address, whatever its open
    /// bytes; none found, or twice, is said and gives none.
    #[test]
    fn a_store_is_found_exactly_once() {
        let profile = Profile::parse(PROFILE).unwrap();
        let (a, b) = (instance(STORE_A, 0x11), instance(STORE_B, 0x22));
        let mut text = vec![0x90u8; 0x30];
        text.extend(&a);
        text.extend([0xcc; 7]);
        let b_at = text.len();
        text.extend(&b);
        text.extend([0x90; 9]);
        let found = locate(&[(0x1_4000_1000, &text)], &profile);
        assert_eq!(
            found,
            [
                Ok(Found {
                    at: 0x1_4000_1000 + 0x30 + 6,
                    len: 5
                }),
                Ok(Found {
                    at: 0x1_4000_1000 + b_at + 6,
                    len: 6
                }),
            ]
        );
        // The second in another section, the first twice (open bytes differ).
        let mut more = text.clone();
        more.extend(instance(STORE_A, 0x77));
        let other = [0x90u8, 0x90];
        let found = locate(
            &[(0x1000, &more), (0x9_0000, &other), (0x5_0000, &b)],
            &profile,
        );
        assert!(
            found[0].as_ref().unwrap_err().contains("matches 2 places"),
            "{found:?}"
        );
        // The second: in two sections now.
        assert!(
            found[1].as_ref().unwrap_err().contains("matches 2 places"),
            "{found:?}"
        );
        // Missing.
        let found = locate(&[(0x1000, &a)], &profile);
        assert_eq!(
            found[0],
            Ok(Found {
                at: 0x1000 + 6,
                len: 5
            })
        );
        assert!(found[1].as_ref().unwrap_err().contains("not in this build"));
        // A changed fixed byte is no match.
        let mut changed = a.clone();
        changed[8] = 0x31;
        assert!(locate(&[(0x1000, &changed)], &profile)[0].is_err());
        // Cut off at the end is no match either.
        assert!(locate(&[(0x1000, &a[..a.len() - 1])], &profile)[0].is_err());
    }

    /// A function of our own in memory of our own, whose store the profile names:
    /// found once in the mapped "image", patched, and run through the stub.
    #[test]
    fn a_store_of_a_mapped_image_is_patched_and_still_stores() {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualAlloc(at: *const u8, size: usize, kind: u32, protect: u32) -> *mut u8;
        }
        // MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE.
        const COMMIT: u32 = 0x3000;
        const RWX: u32 = 0x40;
        // fn(table: rcx, offset: rdx, record: r8): load the record, store it at table + offset.
        const CODE: [u8; 14] = [
            0xc4, 0xc1, 0x7c, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x11, 0xc5, 0xf8, 0x77, 0xc3,
        ];
        let profile =
            Profile::parse("store = 5 5 : C4 C1 7C 10 00 C5 FC 11 04 11 C5 F8 77 C3").unwrap();
        let section = |rva: u32, len: usize| victory_pe::SectionInfo {
            name: ".text".into(),
            rva,
            virtual_size: len as u32,
            file_offset: 0,
            file_size: len as u32,
            characteristics: EXECUTE | 0x4000_0000,
        };
        // SAFETY: fresh memory of our own, never freed (a test).
        let image = unsafe { VirtualAlloc(std::ptr::null(), 0x10000, COMMIT, RWX) } as usize;
        assert_ne!(image, 0);
        let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x6b);
        let run = |at: usize| {
            let mut table = [0u8; 96];
            let f: extern "system" fn(*mut u8, usize, *const u8) =
                unsafe { std::mem::transmute(at) };
            f(table.as_mut_ptr(), 32, record.as_ptr());
            table
        };
        // SAFETY: our own memory; nothing runs the code but the calls below.
        unsafe {
            std::ptr::copy_nonoverlapping(CODE.as_ptr(), (image + 0x100) as *mut u8, CODE.len());
            // Twice in the image: left alone.
            std::ptr::copy_nonoverlapping(CODE.as_ptr(), (image + 0x200) as *mut u8, CODE.len());
            let both = [section(0x100, 0x200), section(0x1000, 0x100)];
            let marks = victory_hle::port_descriptors::marks_address();
            let out = patch_image(image, 0x10000, &both, &profile, marks);
            assert_eq!(out.done, 0, "{out:?}");
            assert!(out.notes[0].contains("matches 2 places"), "{out:?}");
            assert_eq!(*((image + 0x105) as *const u8), 0xc5, "left alone");
            // Once: patched, and the code stores as before.
            let one = [section(0x100, 0x100)];
            let out = patch_image(image, 0x10000, &one, &profile, marks);
            assert_eq!(
                out,
                Patched {
                    done: 1,
                    notes: vec![]
                }
            );
            assert_eq!(*((image + 0x105) as *const u8), 0xe9, "a jump");
            let table = run(image + 0x100);
            assert_eq!(&table[32..64], &record);
            assert!(table[..32].iter().chain(&table[64..]).all(|&b| b == 0));
            // A section outside the image, or not code, is not looked in.
            let (outside, data) = (
                section(0xffff_0000, 0x100),
                victory_pe::SectionInfo {
                    characteristics: 0x4000_0000,
                    ..section(0x100, 0x100)
                },
            );
            assert_eq!(code_of(image, 0x10000, &[outside, data]).len(), 0);
        }
    }

    /// The user's own profile against the user's own title: the executable
    /// (`VICTORY_TITLE_IMAGE`) is read, nothing runs or is written. Skipped
    /// unless both it and `VICTORY_DESCRIPTORS_PROFILE` are set.
    #[test]
    fn a_users_profile_finds_its_stores_in_a_users_title() {
        let (Some(image), Some(profile)) = (
            std::env::var_os("VICTORY_TITLE_IMAGE"),
            std::env::var_os("VICTORY_DESCRIPTORS_PROFILE"),
        ) else {
            return;
        };
        let profile = Profile::parse(&std::fs::read_to_string(profile).unwrap()).unwrap();
        let prepared = victory_loader::prepare(&std::fs::read(image).unwrap()).unwrap();
        println!("build {}", prepared.original_hash.to_hex());
        if let Some(build) = &profile.build {
            assert_eq!(
                build,
                prepared.original_hash.to_hex().as_str(),
                "a profile of another build"
            );
        }
        let sections: Vec<(usize, &[u8])> = prepared
            .info
            .sections
            .iter()
            .filter(|s| s.characteristics & EXECUTE != 0)
            .map(|s| {
                let len = match s.virtual_size {
                    0 => s.file_size,
                    vs => vs.min(s.file_size),
                };
                let from = s.file_offset as usize;
                // The address of a match is its image offset.
                (s.rva as usize, &prepared.bytes[from..from + len as usize])
            })
            .collect();
        let found = locate(&sections, &profile);
        for (n, one) in found.iter().enumerate() {
            println!("store {n}: {one:?}");
        }
        assert!(found.iter().all(Result::is_ok), "{found:?}");
    }
}
