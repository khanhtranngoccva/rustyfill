use experiments::arithmetic_benchmarks;

fn main() {
    let in_range = arithmetic_benchmarks::run_all();
    let overflow = arithmetic_benchmarks::run_overflow_all();
    let workflow = arithmetic_benchmarks::run_workflow_all();
    arithmetic_benchmarks::print_report(&in_range, &overflow, &workflow);
}
