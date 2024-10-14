# Function to calculate the maximum possible score
def max_score(arr, n):
    max_score_val = 0
    
    # Traverse the array and select elements
    for start in range(2):  # Either start at index 0 or 1
        selected = []
        for i in range(start, n, 2):
            selected.append(arr[i])
        
        # Calculate the score for this set of selections
        if selected:
            score = max(selected) + len(selected)
            max_score_val = max(max_score_val, score)
    
    return max_score_val

# Main function to handle multiple test cases
def main():
    t = int(input())  # Number of test cases
    
    for _ in range(t):
        n = int(input())  # Length of the array
        arr = list(map(int, input().split()))  # Array of integers
        
        # Get the maximum possible score
        result = max_score(arr, n)
        
        # Output the result for the current test case
        print(result)

# Call the main function to execute the program
if __name__ == "__main__":
    main()
