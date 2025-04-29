import sys
from latex_eval import (
    evaluate_latex, 
    solve_equation, 
    solve_system_of_equations,
    factor_expression,
    expand_expression,
    simplify_expression,
    display_evaluation
)

# Test the evaluator with error handling
def test_evaluator():
    print("===== Basic LaTeX Evaluator Tests =====\n")
    
    tests = [
        ("Basic arithmetic", "2 + 3 * 4"),
        ("Fractions", "\\frac{3}{4} + \\frac{1}{2}"),
        ("Square roots", "\\sqrt{16} + \\sqrt{9}"),
        ("Exponents", "2^{4} + 3^{2}"),
        ("Variables", "x^2 + 2x + 1"),
        ("Mixed operations", "\\frac{1}{2} \\cdot (x^2 + 4) + \\sqrt{16}"),
        ("Complex expression", "(x+1)^2 - \\frac{x^2 - 4}{2x+3} + \\sqrt{x+4}")
    ]
    
    # Test evaluation
    for name, expr in tests:
        print(f"Test: {name}")
        print(f"Expression: {expr}")
        
        try:
            # Parse and get symbolic form
            parsed = evaluate_latex(expr, numerical=False)
            print(f"Parsed result: {parsed}")
            
            # Try with x = 2
            if 'x' in str(parsed):
                numerical = evaluate_latex(expr, subs={'x': 2})
                print(f"With x = 2: {numerical}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

# Test equation solving with error handling
def test_equation_solver():
    print("===== Equation Solver Tests =====\n")
    
    equations = [
        ("Linear equation", "2x + 3 = 7"),
        ("Quadratic equation", "x^2 - 4 = 0"),
        ("Cubic equation", "x^3 - 6x^2 + 11x - 6 = 0"),
        ("Fractional equation", "\\frac{x+1}{x-1} = 2"),
        ("Equation with square root", "\\sqrt{x} = 3")
    ]
    
    for name, eq in equations:
        print(f"Test: {name}")
        print(f"Equation: {eq}")
        
        try:
            # Solve the equation
            solutions = solve_equation(eq)
            print(f"Solutions: {solutions}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

# Test system of equations solver with error handling
def test_system_solver():
    print("===== System of Equations Solver Tests =====\n")
    
    systems = [
        ("2x2 linear system", ["2x + y = 5", "x - y = 1"]),
        ("3x3 linear system", ["x + y + z = 6", "2x - y + z = 3", "x + 2y - z = 0"])
    ]
    
    for name, eqs in systems:
        print(f"Test: {name}")
        print(f"Equations: {eqs}")
        
        try:
            # Solve the system
            solution = solve_system_of_equations(eqs)
            print(f"Solution: {solution}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

# Test algebraic manipulations with error handling
def test_algebraic_operations():
    print("===== Algebraic Operations Tests =====\n")
    
    operations = [
        ("Factoring", "x^2 - 4", factor_expression),
        ("Factoring quadratic", "x^2 + 5x + 6", factor_expression),
        ("Expanding", "(x+1)(x+2)", expand_expression),
        ("Expanding cubic", "(x+1)^3", expand_expression),
        ("Simplifying", "\\frac{x^2-1}{x-1}", simplify_expression),
        ("Simplifying complex fraction", "\\frac{\\frac{1}{x} + \\frac{1}{y}}{\\frac{1}{x^2} - \\frac{1}{y^2}}", simplify_expression)
    ]
    
    for name, expr, func in operations:
        print(f"Test: {name}")
        print(f"Expression: {expr}")
        
        try:
            # Apply the operation
            result = func(expr)
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    # Run all tests or specific tests based on command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "evaluate":
            test_evaluator()
        elif test_name == "solve":
            test_equation_solver()
        elif test_name == "system":
            test_system_solver()
        elif test_name == "algebra":
            test_algebraic_operations()
        else:
            print(f"Unknown test: {test_name}")
    else:
        # Run all tests
        test_evaluator()
        test_equation_solver()
        test_system_solver()
        test_algebraic_operations()