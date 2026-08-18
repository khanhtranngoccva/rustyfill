use experiments::arithmetic_benchmarks;

fn main() {
    let results = arithmetic_benchmarks::run_all();
    arithmetic_benchmarks::print_report(&results);
}
