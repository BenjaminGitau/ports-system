import requests
from threading import Thread
import time
from datetime import datetime
import random
import string
import json

# Configuration
BASE_URL = "http://localhost:5000"  # Update if your app runs on a different URL
NUM_THREADS = 50  # Number of concurrent users to simulate
TEST_DURATION = 60  # Duration of the test in seconds

# Test user credentials
TEST_USER = {
    "email": "test@example.com",
    "password": "password"
}

# Generate random strings for testing
def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Worker function for each simulated user
def worker(user_id, results):
    session = requests.Session()
    user_stats = {
        "requests": 0,
        "success": 0,
        "errors": 0,
        "response_times": []
    }
    
    # Register a new user (simulating different users)
    user_data = {
        "email": f"stress_{user_id}@test.com",
        "password": "password",
        "first_name": f"Stress{user_id}",
        "last_name": "User",
        "national_id": str(10000000 + user_id)
    }
    
    try:
        # Register
        start_time = time.time()
        response = session.post(f"{BASE_URL}/register", data=user_data)
        user_stats["requests"] += 1
        if response.status_code == 200:
            user_stats["success"] += 1
        else:
            user_stats["errors"] += 1
        user_stats["response_times"].append(time.time() - start_time)
        
        # Login
        start_time = time.time()
        response = session.post(f"{BASE_URL}/login", data={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        user_stats["requests"] += 1
        if response.status_code == 200:
            user_stats["success"] += 1
        else:
            user_stats["errors"] += 1
        user_stats["response_times"].append(time.time() - start_time)
        
        # Simulate user activity for the test duration
        end_time = time.time() + TEST_DURATION
        while time.time() < end_time:
            # Randomly select an endpoint to test
            endpoint = random.choice([
                ("/", "GET", None),
                ("/detections", "GET", None),
                ("/camera/status", "GET", None),
                ("/camera/toggle", "POST", None),
                ("/chat/messages", "GET", None),
                ("/chat/send", "POST", {"message": f"Test message {random_string()}"}),
                ("/active_users", "GET", None)
            ])
            
            try:
                start_time = time.time()
                if endpoint[1] == "GET":
                    response = session.get(f"{BASE_URL}{endpoint[0]}")
                else:
                    response = session.post(f"{BASE_URL}{endpoint[0]}", data=endpoint[2])
                
                user_stats["requests"] += 1
                if response.status_code == 200:
                    user_stats["success"] += 1
                else:
                    user_stats["errors"] += 1
                user_stats["response_times"].append(time.time() - start_time)
                
                # Random delay between requests (0.1-1 second)
                time.sleep(random.uniform(0.1, 1))
                
            except Exception as e:
                user_stats["errors"] += 1
                user_stats["requests"] += 1
                print(f"User {user_id} error: {str(e)}")
                
    except Exception as e:
        print(f"User {user_id} setup failed: {str(e)}")
    
    results[user_id] = user_stats

# Main test function
def run_stress_test():
    print(f"Starting stress test with {NUM_THREADS} users for {TEST_DURATION} seconds...")
    start_time = datetime.now()
    
    threads = []
    results = {}
    
    # Create and start worker threads
    for i in range(NUM_THREADS):
        t = Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Calculate statistics
    total_requests = 0
    total_success = 0
    total_errors = 0
    all_response_times = []
    
    for user_id, stats in results.items():
        total_requests += stats["requests"]
        total_success += stats["success"]
        total_errors += stats["errors"]
        all_response_times.extend(stats["response_times"])
    
    if all_response_times:
        avg_response_time = sum(all_response_times) / len(all_response_times)
        min_response_time = min(all_response_times)
        max_response_time = max(all_response_times)
    else:
        avg_response_time = min_response_time = max_response_time = 0
    
    test_duration = (datetime.now() - start_time).total_seconds()
    requests_per_second = total_requests / test_duration if test_duration > 0 else 0
    
    # Print summary
    print("\nStress Test Results:")
    print(f"Test Duration: {test_duration:.2f} seconds")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {total_success}")
    print(f"Failed Requests: {total_errors}")
    print(f"Requests per Second: {requests_per_second:.2f}")
    print(f"Average Response Time: {avg_response_time:.4f} seconds")
    print(f"Minimum Response Time: {min_response_time:.4f} seconds")
    print(f"Maximum Response Time: {max_response_time:.4f} seconds")
    
    # Save detailed results to file
    with open("stress_test_results.json", "w") as f:
        json.dump({
            "summary": {
                "test_duration": test_duration,
                "total_requests": total_requests,
                "successful_requests": total_success,
                "failed_requests": total_errors,
                "requests_per_second": requests_per_second,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "detailed_results": results
        }, f, indent=2)
    
    print("Detailed results saved to stress_test_results.json")

if __name__ == "__main__":
    run_stress_test()