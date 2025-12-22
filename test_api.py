"""
Quick test script to verify the AI service
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_anomaly_detection():
    """Test anomaly detection endpoint"""
    print("\n" + "="*60)
    print("Testing Anomaly Detection")
    print("="*60)
    
    # Test with normal composition
    normal_comp = {
        "composition": {
            "Fe": 85.5,
            "C": 3.2,
            "Si": 2.1,
            "Mn": 0.6,
            "P": 0.04,
            "S": 0.02
        }
    }
    
    print("\nTest 1: Normal Composition")
    print(f"Input: {json.dumps(normal_comp, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/anomaly/predict", json=normal_comp)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test with deviated composition
    deviated_comp = {
        "composition": {
            "Fe": 81.2,
            "C": 4.8,
            "Si": 3.5,
            "Mn": 0.3,
            "P": 0.12,
            "S": 0.05
        }
    }
    
    print("\nTest 2: Deviated Composition")
    print(f"Input: {json.dumps(deviated_comp, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/anomaly/predict", json=deviated_comp)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_alloy_correction():
    """Test alloy correction endpoint"""
    print("\n" + "="*60)
    print("Testing Alloy Correction")
    print("="*60)
    
    # Test with deviated SG-IRON
    request_data = {
        "grade": "SG-IRON",
        "composition": {
            "Fe": 81.2,
            "C": 4.4,
            "Si": 3.1,
            "Mn": 0.4,
            "P": 0.05,
            "S": 0.02
        }
    }
    
    print("\nTest: SG-IRON Correction")
    print(f"Input: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/alloy/recommend", json=request_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_grades():
    """Test grades endpoint"""
    print("\n" + "="*60)
    print("Testing Grades Endpoint")
    print("="*60)
    
    # Get all grades
    print("\nFetching all grades...")
    response = requests.get(f"{BASE_URL}/grades")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Get specific grade
    print("\nFetching SG-IRON specification...")
    response = requests.get(f"{BASE_URL}/grades/SG-IRON")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" METALLISENSE AI SERVICE - API TESTS")
    print("="*70)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the service is running: python app/main.py")
    print("")
    
    try:
        # Run tests
        tests = [
            ("Health Check", test_health),
            ("Anomaly Detection", test_anomaly_detection),
            ("Alloy Correction", test_alloy_correction),
            ("Grades API", test_grades)
        ]
        
        results = {}
        for name, test_func in tests:
            try:
                success = test_func()
                results[name] = "✓ PASSED" if success else "✗ FAILED"
            except requests.exceptions.ConnectionError:
                results[name] = "✗ CONNECTION ERROR"
                print(f"\nError: Cannot connect to {BASE_URL}")
                print("Make sure the API service is running!")
                break
            except Exception as e:
                results[name] = f"✗ ERROR: {str(e)}"
        
        # Summary
        print("\n" + "="*70)
        print(" TEST SUMMARY")
        print("="*70)
        
        for test_name, result in results.items():
            print(f"{test_name:.<40} {result}")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")

if __name__ == "__main__":
    main()
