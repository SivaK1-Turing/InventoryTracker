def _display_results(self, 
                    results: Dict[str, List[Tuple[bool, str]]], 
                    verbose: bool = False) -> int:
    """
    Display notification results and return exit code.
    
    Args:
        results: Dictionary mapping notifier names to lists of (success, message) tuples
        verbose: Whether to display detailed results
        
    Returns:
        Exit code (0 for all success, 1 for any failures)
    """
    total_success = 0
    total_failure = 0
    
    # Process the results - count successes and failures
    for notifier_name, result_list in results.items():
        for success, message in result_list:
            if success:
                total_success += 1
            else:
                total_failure += 1
    
    print("\nNotification Summary:")
    
    # Show overall status with color coding
    if total_failure == 0 and total_success > 0:
        print(self.color.green(f"✓ All notifications sent successfully ({total_success} total)"))
        result_code = 0
    elif total_success > 0 and total_failure > 0:
        print(self.color.yellow(f"⚠ Partial success: {total_success} successful, {total_failure} failed"))
        result_code = 1
    else:
        print(self.color.red(f"✗ All notifications failed ({total_failure} total)"))
        result_code = 1
        
    # Display detailed results if requested
    if verbose:
        print("\nDetailed Results:")
        for notifier_name, result_list in results.items():
            print(f"\n{notifier_name}:")
            for success, message in result_list:
                status_symbol = self.color.green("✓") if success else self.color.red("✗")
                print(f"  {status_symbol} {message}")
    
    # Return appropriate exit code
    return result_code