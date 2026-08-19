use experiments::display_allocation_tests;
use experiments::display_allocation_tests::TestResult;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Child mode: a single string argument selects one test by ID.
    if args.len() == 2 {
        display_allocation_tests::run_single_test(&args[1]);
        std::process::exit(0);
    }

    // Runner mode: iterate all tests, spawning a child for each.
    let tests = display_allocation_tests::all_tests();
    let mut results: Vec<(&str, &str, TestResult)> = Vec::new();

    for test in tests.iter() {
        print!("[{:<25}] testing {:<45} ... ", test.id, test.name);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let result = display_allocation_tests::run_in_child(test.id);
        results.push((test.id, test.name, result.clone()));
        match &result {
            TestResult::Safe => println!("\u{2714} SAFE"),
            TestResult::ImplicitlyAllocates => println!("\u{2716} IMPLICITLY ALLOCATES"),
            TestResult::Error(e) => println!("\u{26A0} ERROR: {}", e),
        }
    }

    // Summary table
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));

    let safe: usize = results
        .iter()
        .filter(|(_, _, r)| *r == TestResult::Safe)
        .count();
    let allocates: usize = results
        .iter()
        .filter(|(_, _, r)| *r == TestResult::ImplicitlyAllocates)
        .count();
    let errors: usize = results
        .iter()
        .filter(|(_, _, r)| matches!(r, TestResult::Error(_)))
        .count();

    for (id, name, result) in &results {
        let icon = match result {
            TestResult::Safe => "\u{2714}",
            TestResult::ImplicitlyAllocates => "\u{2716}",
            TestResult::Error(_) => "\u{26A0}",
        };
        let label = match result {
            TestResult::Safe => "SAFE",
            TestResult::ImplicitlyAllocates => "IMPLICITLY ALLOCATES",
            TestResult::Error(e) => &format!("ERROR ({})", e),
        };
        println!("[{:<25}] {} {:<45} {}", id, icon, name, label);
    }

    println!("{}", "-".repeat(80));
    println!(
        "safe: {} | allocates: {} | errors: {}",
        safe, allocates, errors
    );
    println!("{}", "=".repeat(80));
    println!();
    println!("Run a single test with: cargo run -p display-allocation-tests -- <id>");

    if errors > 0 {
        std::process::exit(1);
    }
}
