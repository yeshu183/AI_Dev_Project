import re
import math
import sympy as sp
from sympy import symbols, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def parse_latex_to_sympy(latex_expr):
    """
    Convert a LaTeX mathematical expression to a SymPy expression.
    
    Args:
        latex_expr (str): LaTeX expression to convert
        
    Returns:
        sympy expression: The converted expression
    """
    # Define basic symbols
    x, y, z = symbols('x y z')
    
    # Import sqrt and other functions from sympy at the start
    from sympy import sqrt, log, pi, E
    
    # Replace LaTeX constructs with Python/SymPy equivalents
    expr = latex_expr
    
    # Handle fractions: \frac{num}{den}
    expr = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', expr)
    
    # Handle square roots: \sqrt{expr}
    expr = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', expr)
    
    # Handle nth roots: \sqrt[n]{expr}
    expr = re.sub(r'\\sqrt\[([^{}]*)\]\{([^{}]*)\}', r'((\2)**(1/(\1)))', expr)
    
    # Handle logarithms
    expr = expr.replace('\\ln', 'sp.log')
    expr = expr.replace('\\log', 'sp.log')
    
    # Handle constants
    expr = expr.replace('\\pi', 'sp.pi')
    expr = expr.replace('\\e', 'sp.E')
    
    # Handle exponents with braces: x^{expr}
    expr = re.sub(r'\^(\{[^{}]*\})', lambda m: f'**{m.group(1)}', expr)
    
    # Handle simple exponents: x^2
    expr = re.sub(r'\^(\d+)', r'**\1', expr)
    
    # Handle multiplication symbols
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\times', '*')
    expr = expr.replace('\\div', '/')
    
    # Handle implicit multiplication (e.g., 2x -> 2*x)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    
    # Remove curly braces from remaining expressions
    expr = re.sub(r'\{([^{}]*)\}', r'\1', expr)
    
    try:
        # Set up transformations for the parser
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        # Convert the cleaned expression to a SymPy object
        return parse_expr(expr, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Failed to parse LaTeX expression: {e}")

def evaluate_latex(latex_expr, numerical=True, subs=None):
    """
    Evaluate a LaTeX mathematical expression.
    
    Args:
        latex_expr (str): LaTeX expression to evaluate
        numerical (bool): If True, return a numerical result, otherwise return symbolic
        subs (dict): Dictionary of variable substitutions, e.g., {'x': 2}
        
    Returns:
        The evaluated result (float or sympy expression)
    """
    try:
        sympy_expr = parse_latex_to_sympy(latex_expr)
        
        if subs:
            sympy_expr = sympy_expr.subs(subs)
            
        if numerical:
            try:
                return float(sympy_expr.evalf())
            except:
                # Return the evalf() result if it can't be converted to float
                return sympy_expr.evalf()
        else:
            return sympy_expr
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

def solve_equation(latex_expr, var='x'):
    """
    Solve an equation expressed in LaTeX.
    
    Args:
        latex_expr (str): LaTeX equation (containing =)
        var (str): Variable to solve for
        
    Returns:
        list: Solutions to the equation
    """
    # Split by equals sign
    if '=' not in latex_expr:
        raise ValueError("No equation detected. Equation must contain '='")
        
    parts = latex_expr.split('=')
    if len(parts) != 2:
        raise ValueError("Invalid equation format. Must have exactly one '='")
        
    left_expr = parse_latex_to_sympy(parts[0])
    right_expr = parse_latex_to_sympy(parts[1])
    
    # Move everything to left side: left - right = 0
    equation = left_expr - right_expr
    
    # Solve
    symbol = sp.Symbol(var)
    solutions = sp.solve(equation, symbol)
    
    return solutions

def solve_system_of_equations(equations, variables=None):
    """
    Solve a system of equations expressed in LaTeX.
    
    Args:
        equations (list): List of LaTeX equation strings (each containing =)
        variables (list): List of variables to solve for. If None, will be inferred.
        
    Returns:
        dict: Solutions to the system of equations
    """
    if not variables:
        # Infer variables, default to x, y, z, etc.
        default_vars = ['x', 'y', 'z', 'w', 'u', 'v', 't', 's']
        variables = default_vars[:len(equations)]
    
    # Parse each equation
    parsed_eqs = []
    for eq in equations:
        if '=' not in eq:
            raise ValueError(f"No equation detected in '{eq}'. Equation must contain '='")
            
        parts = eq.split('=')
        if len(parts) != 2:
            raise ValueError(f"Invalid equation format in '{eq}'. Must have exactly one '='")
            
        left_expr = parse_latex_to_sympy(parts[0])
        right_expr = parse_latex_to_sympy(parts[1])
        
        # Move everything to left side: left - right = 0
        parsed_eqs.append(left_expr - right_expr)
    
    # Create symbols for variables
    symbols_list = [sp.Symbol(var) for var in variables]
    
    # Solve the system
    solution = sp.solve(parsed_eqs, symbols_list)
    
    return solution

def factor_expression(latex_expr):
    """
    Factor a polynomial expression.
    
    Args:
        latex_expr (str): LaTeX expression to factor
        
    Returns:
        sympy expression: The factored expression
    """
    expr = parse_latex_to_sympy(latex_expr)
    return sp.factor(expr)

def expand_expression(latex_expr):
    """
    Expand a factored expression.
    
    Args:
        latex_expr (str): LaTeX expression to expand
        
    Returns:
        sympy expression: The expanded expression
    """
    expr = parse_latex_to_sympy(latex_expr)
    return sp.expand(expr)

def simplify_expression(latex_expr):
    """
    Simplify an expression.
    
    Args:
        latex_expr (str): LaTeX expression to simplify
        
    Returns:
        sympy expression: The simplified expression
    """
    expr = parse_latex_to_sympy(latex_expr)
    return sp.simplify(expr)

def latex_to_string(sympy_expr):
    """
    Convert a SymPy expression back to a LaTeX string.
    
    Args:
        sympy_expr: SymPy expression
        
    Returns:
        str: LaTeX representation of the expression
    """
    return sp.latex(sympy_expr)

def display_evaluation(latex_expr, subs=None):
    """
    Display the steps of evaluating a LaTeX expression.
    
    Args:
        latex_expr (str): LaTeX expression to evaluate
        subs (dict): Dictionary of variable substitutions
    
    Returns:
        dict: A dictionary containing the original expression, 
              its parsed form, symbolic result, and numerical result
    """
    try:
        # Parse LaTeX to SymPy
        sympy_expr = parse_latex_to_sympy(latex_expr)
        
        # Get symbolic form
        if subs:
            symbolic_result = sympy_expr.subs(subs)
            substituted_latex = latex_to_string(symbolic_result)
        else:
            symbolic_result = sympy_expr
            substituted_latex = latex_to_string(sympy_expr)
        
        # Get numerical result
        try:
            numerical_result = symbolic_result.evalf()
        except:
            numerical_result = "Cannot evaluate numerically"
            
        result = {
            'original_latex': latex_expr,
            'parsed_expr': str(sympy_expr),
            'symbolic_result': str(symbolic_result),
            'symbolic_latex': substituted_latex,
            'numerical_result': numerical_result
        }
        
        return result
    except Exception as e:
        return {
            'original_latex': latex_expr,
            'error': str(e)
        }

# Examples for testing
if __name__ == "__main__":
    # Test basic operations
    print("Example 1: Basic fraction addition")
    result = display_evaluation("\\frac{3}{4} + \\frac{1}{2}")
    print(f"Original: {result['original_latex']}")
    print(f"Parsed: {result['parsed_expr']}")
    print(f"Symbolic: {result['symbolic_result']}")
    print(f"LaTeX: {result['symbolic_latex']}")
    print(f"Numerical: {result['numerical_result']}")
    print()
    
    # Test with variables and substitutions
    print("Example 2: Expression with variables")
    result = display_evaluation("x^2 + 2x + 1", {'x': 3})
    print(f"Original: {result['original_latex']}")
    print(f"With x=3: {result['numerical_result']}")
    print()
    
    # Test equation solving
    print("Example 3: Solving an equation")
    solutions = solve_equation("x^2 - 4 = 0")
    print(f"Equation: x^2 - 4 = 0")
    print(f"Solutions: {solutions}")
    print()
    
    # Test system of equations
    print("Example 4: Solving a system of equations")
    eqs = ["2x + y = 5", "x - y = 1"]
    solution = solve_system_of_equations(eqs)
    print(f"Equations: {eqs}")
    print(f"Solution: {solution}")
    print()
    
    # Test factoring
    print("Example 5: Factoring a polynomial")
    expr = "x^2 - 4"
    factored = factor_expression(expr)
    print(f"Expression: {expr}")
    print(f"Factored: {factored}")
    print()
    
    # Test expansion
    print("Example 6: Expanding an expression")
    expr = "(x+1)(x+2)"
    expanded = expand_expression(expr)
    print(f"Expression: {expr}")
    print(f"Expanded: {expanded}")
    print()
    
    # Test sqrt and exponents
    print("Example 7: Square roots and exponents")
    result = display_evaluation("\\sqrt{16} + 2^{4}")
    print(f"Original: {result['original_latex']}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Numerical: {result['numerical_result']}")