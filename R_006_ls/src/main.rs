use std::fs;
use std::env::args;

/// Copy of the linux `ls` command (not all the functionality)
/// Usage:
///     ls [OPTION] [PATH]
/// 
/// -a, --all 
///     do not ignore entries starting with .
/// 
/// -f 
///     disable sort
/// 
/// -1
///     show one file per line
fn main() {

    let args_vec: Vec<String> = args().skip(1).collect();

    let all = args_vec.contains(&String::from("-a")) || args_vec.contains(&String::from("--all"));
    let sort = args_vec.contains(&String::from("-f"));
    let seperator = if args_vec.contains(&String::from("-1")) { "\n" } else { " " };
    let file: Vec<String> = args().skip(1).filter(|x| !x.contains('-')).collect();
    let file = if file.is_empty() { "./" } else { &file[0] };

    // Read all files and folders in current dir
    let mut strings: Vec<_> = fs::read_dir(file)
        .expect("Can't access path")
        .map(|x| x.unwrap().file_name().into_string().unwrap())
        .collect();
    
    if sort { strings.sort(); };
    let strings = strings.into_iter()
        .filter(|x| x.as_str().chars().nth(0).unwrap() != '.' || all);

    for string in strings { 
        print!("{}{}", string, seperator); 
    };
    println!("");
}