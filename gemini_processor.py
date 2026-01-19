"""
Gemini API processor for extracting MBTI personality data from images.
"""
import os
import base64
import json
from typing import Dict, Optional, Tuple
from google import genai
from google.genai import types


def process_single_image_with_gemini(
    image_base64: str, gemini_api_key: str, image_index: int = 0
) -> Dict:
    """
    Process a single image with Gemini API to extract personality test data.
    
    Args:
        image_base64: Base64-encoded image
        gemini_api_key: Gemini API key
        image_index: Index of the image being processed (for logging)
        
    Returns:
        Dictionary with:
        - personalityData: Dict with personality traits or None
        - success: Boolean indicating if personality data was found
    """
    if not gemini_api_key:
        print('[Gemini] API key not provided')
        return {'personalityData': None, 'success': False}
    
    print(f'[Gemini] Processing image {image_index} for personality data')
    
    try:
        # Create client
        client = genai.Client(api_key=gemini_api_key)
        
        # Prepare the prompt
        gemini_prompt = """You are given an image. Your task is to extract personality test results from it.

CRITICAL REQUIREMENTS:
1. Examine the image carefully. It MAY contain a personality test with 5 traits showing percentages.
2. The image may be completely unrelated (wrong image) - if so, return success: false.
3. The test results MAY be incomplete (missing percentage values). If so, try to extract only personality type.
4. Personality type is usually represented by a (4+1)-letter code (e.g., ESFJ-T, ENTJ-A). If you can clearly identify this code, include it in the response.


WHAT TO LOOK FOR:
A complete personality test image must show ALL of the following:
- Exactly 5 personality traits with VISIBLE percentage values
- The 5 traits should be: 
  * Extraverted or Introverted (Energy)
  * Intuitive or Observant (Mind)
  * Thinking vs Feeling (Nature)
  * Judging vs Prospecting (Tactics)
  * Assertive vs Turbulent (Identity)
- Each trait must have a percentage number clearly displayed (0-100)
An incomplete test may show only the 5-letter code (e.g., ESFJ-T) without percentages.

YOUR TASK:
1. Determine if this image contains a COMPLETE personality test with all 5 traits and their percentages, or just a 5-letter personality type code without percentages.
2. If complete result is found, extract the EXACT percentage values shown in the image (do NOT estimate or assume).
3. Extract the dominant trait letter for each trait (E/I for Energy, N/S for Mind, T/F for Nature, J/P for Tactics, A/T for Identity).
4. If only the personality type code is found, extract the 5-letter code and map it to the corresponding 5 traits. Set percentage values to 0.
5. If neither complete traits with percentages nor personality type code is found, return success: false.

IMPORTANT RULES:

- Set success to TRUE only if ALL 5 traits with percentages are clearly visible and extractable, or a clearly identifiable 5-letter personality type code is found.
- Set success to FALSE if:
  * The image does not contain personality test results (neither percentages nor type code)
  * The percentages are not readable or visible in a complete test result
  * Less than 5 traits are shown and no personality type code is found
  * The trait labels don't match the expected personality traits
- Do NOT assume, estimate, or guess any values - extract only what you can clearly see
- Return the actual percentage numbers shown in the image, not approximations"""
        
        # Decode base64 image to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Create inline data part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )
        
        # Configure generation with JSON schema
        response_schema = {
            'type': 'OBJECT',
            'properties': {
                'success': {'type': 'BOOLEAN'},
                'energy': {'type': 'NUMBER'},
                'energy_type': {'type': 'STRING'},
                'mind': {'type': 'NUMBER'},
                'mind_type': {'type': 'STRING'},
                'nature': {'type': 'NUMBER'},
                'nature_type': {'type': 'STRING'},
                'tactics': {'type': 'NUMBER'},
                'tactics_type': {'type': 'STRING'},
                'identity': {'type': 'NUMBER'},
                'identity_type': {'type': 'STRING'}
            },
            'required': [
                'success', 'energy', 'energy_type',
                'mind', 'mind_type', 'nature', 'nature_type',
                'tactics', 'tactics_type', 'identity', 'identity_type'
            ]
        }
        
        # Add JSON format instruction to the prompt
        gemini_prompt += """

RESPONSE FORMAT:
Return your response as a valid JSON object with this exact structure:
{
    "success": true/false,
    "energy": percentage_number,
    "energy_type": "E" or "I",
    "mind": percentage_number,
    "mind_type": "N" or "S",
    "nature": percentage_number,
    "nature_type": "T" or "F",
    "tactics": percentage_number,
    "tactics_type": "J" or "P",
    "identity": percentage_number,
    "identity_type": "A" or "T"
}

If success is false, still provide the other fields but they can be null or 0."""
        
        # Generate content without JSON mode
        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=[gemini_prompt, image_part]
        )
        
        # Parse response
        if response and response.text:
            try:
                # Try to extract JSON from the response text
                response_text = response.text.strip()
                
                # Look for JSON in the response (might be wrapped in markdown)
                if '```json' in response_text:
                    # Extract JSON from markdown code block
                    start = response_text.find('```json') + 7
                    end = response_text.find('```', start)
                    if end != -1:
                        response_text = response_text[start:end].strip()
                elif '{' in response_text and '}' in response_text:
                    # Find the JSON object in the text
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    response_text = response_text[start:end]
                
                data = json.loads(response_text)
                
                if data.get('success') is True:
                    print(f'[Gemini] Success: Personality test found in image {image_index}')
                    return {
                        'personalityData': data,
                        'success': True
                    }
                else:
                    print(f'[Gemini] No personality test found in image {image_index}')
                    return {'personalityData': None, 'success': False}
                    
            except json.JSONDecodeError as e:
                print(f'[Gemini] Failed to parse JSON response: {e}')
                print(f'[Gemini] Raw response: {response.text[:200]}...')
                return {'personalityData': None, 'success': False}
        else:
            print('[Gemini] No response from Gemini')
    
    except Exception as e:
        error_msg = str(e)
        print(f'[Gemini] Error processing image {image_index}: {error_msg}')
        
        # Check for specific error types
        error_lower = error_msg.lower()
        
        # Rate limit / quota exceeded
        if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_lower:
            print('[Gemini] Rate limit exceeded - returning error to caller')
            return {
                'personalityData': None, 
                'success': False,
                'error': 'rate_limit_exceeded',
                'error_message': error_msg
            }
        
        # Authentication errors
        if '401' in error_msg or 'UNAUTHENTICATED' in error_msg or 'invalid api key' in error_lower or 'api_key_invalid' in error_lower:
            print('[Gemini] Authentication failed - invalid API key')
            return {
                'personalityData': None,
                'success': False,
                'error': 'invalid_api_key',
                'error_message': error_msg
            }
        
        # Permission denied
        if '403' in error_msg or 'PERMISSION_DENIED' in error_msg or 'permission denied' in error_lower:
            print('[Gemini] Permission denied - API key lacks required permissions')
            return {
                'personalityData': None,
                'success': False,
                'error': 'permission_denied',
                'error_message': error_msg
            }
        
        # Service unavailable
        if '503' in error_msg or 'UNAVAILABLE' in error_msg or 'service unavailable' in error_lower:
            print('[Gemini] Service temporarily unavailable')
            return {
                'personalityData': None,
                'success': False,
                'error': 'service_unavailable',
                'error_message': error_msg
            }
        
        # Safety filter blocked
        if 'SAFETY' in error_msg or 'blocked' in error_lower or 'safety' in error_lower:
            print('[Gemini] Content blocked by safety filters')
            return {
                'personalityData': None,
                'success': False,
                'error': 'safety_blocked',
                'error_message': error_msg
            }
        
        # Generic error (don't stop processing for these)
        import traceback
        traceback.print_exc()
    
    return {'personalityData': None, 'success': False}


def calculate_tritype(pie_chart_result: Dict) -> Tuple[int, Optional[int]]:
    """
    Calculate tritype values from pie chart segments.
    
    Args:
        pie_chart_result: Dictionary with segment_1 through segment_9 values
        
    Returns:
        Tuple of (tritype_with_8, tritype_without_8)
    """
    # Extract segment values
    triad1_values = [
        {'segment': 8, 'value': pie_chart_result.get('segment_8', 0)},
        {'segment': 9, 'value': pie_chart_result.get('segment_9', 0)},
        {'segment': 1, 'value': pie_chart_result.get('segment_1', 0)}
    ]
    triad2_values = [
        {'segment': 2, 'value': pie_chart_result.get('segment_2', 0)},
        {'segment': 3, 'value': pie_chart_result.get('segment_3', 0)},
        {'segment': 4, 'value': pie_chart_result.get('segment_4', 0)}
    ]
    triad3_values = [
        {'segment': 5, 'value': pie_chart_result.get('segment_5', 0)},
        {'segment': 6, 'value': pie_chart_result.get('segment_6', 0)},
        {'segment': 7, 'value': pie_chart_result.get('segment_7', 0)}
    ]
    
    # Get max segment from each triad (with segment 8)
    def get_max_segment(triad_values):
        return max(triad_values, key=lambda x: x['value'])['segment']
    
    max_segment1_with_8 = get_max_segment(triad1_values)
    max_segment2_with_8 = get_max_segment(triad2_values)
    max_segment3_with_8 = get_max_segment(triad3_values)
    
    # Sort by value to get tritype
    segments_with_8 = [
        {'segment': max_segment1_with_8, 'value': next(t['value'] for t in triad1_values if t['segment'] == max_segment1_with_8)},
        {'segment': max_segment2_with_8, 'value': next(t['value'] for t in triad2_values if t['segment'] == max_segment2_with_8)},
        {'segment': max_segment3_with_8, 'value': next(t['value'] for t in triad3_values if t['segment'] == max_segment3_with_8)}
    ]
    segments_with_8.sort(key=lambda x: x['value'], reverse=True)
    tritype_with_8 = (segments_with_8[0]['segment'] * 100 + 
                     segments_with_8[1]['segment'] * 10 + 
                     segments_with_8[2]['segment'])
    
    # Get max segment from each triad (without segment 8)
    def get_max_segment_without_8(triad_values):
        sorted_vals = sorted(triad_values, key=lambda x: x['value'], reverse=True)
        if sorted_vals[0]['segment'] == 8:
            return sorted_vals[1]['segment']
        return sorted_vals[0]['segment']
    
    max_segment1_without_8 = get_max_segment_without_8(triad1_values)
    max_segment2_without_8 = get_max_segment_without_8(triad2_values)
    max_segment3_without_8 = get_max_segment_without_8(triad3_values)
    
    segments_without_8 = [
        {'segment': max_segment1_without_8, 'value': next(t['value'] for t in triad1_values if t['segment'] == max_segment1_without_8)},
        {'segment': max_segment2_without_8, 'value': next(t['value'] for t in triad2_values if t['segment'] == max_segment2_without_8)},
        {'segment': max_segment3_without_8, 'value': next(t['value'] for t in triad3_values if t['segment'] == max_segment3_without_8)}
    ]
    segments_without_8.sort(key=lambda x: x['value'], reverse=True)
    tritype_without_8 = (segments_without_8[0]['segment'] * 100 + 
                         segments_without_8[1]['segment'] * 10 + 
                         segments_without_8[2]['segment'])
    
    # Check if segment 8 is present in tritype
    has_segment_8 = (max_segment1_with_8 == 8 or 
                     max_segment2_with_8 == 8 or 
                     max_segment3_with_8 == 8)
    
    return tritype_with_8, tritype_without_8 if has_segment_8 else None
