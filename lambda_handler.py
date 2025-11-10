"""
AWS Lambda handler for signal generation.
Deploy this to AWS Lambda with EventBridge (CloudWatch Events) trigger every 3 minutes.
"""
import json
from signal_generator import main as generate_signals

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    
    To deploy:
    1. Zip all files: zip -r function.zip .
    2. Upload to AWS Lambda
    3. Set environment variables: POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY
    4. Create EventBridge rule: rate(3 minutes)
    5. Set memory to 512MB, timeout to 5 minutes
    """
    try:
        print("Starting signal generation...")
        generate_signals()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Signals generated successfully',
                'timestamp': str(event.get('time', 'N/A'))
            })
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

