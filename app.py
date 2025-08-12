from flask import Flask, render_template, request, redirect, jsonify
from tools import *
from datetime import datetime
from qamodel2 import generate_answer
from gemini import improve_agent_details
app = Flask(__name__)

@app.route('/agentregister', methods=['POST'])
def agent_register():
    """
    Endpoint to register an agent by uploading a text file.
    Saves both the original and Gemini-improved versions.
    Returns: JSON with agent_id and status.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .txt files are allowed'}), 400

        # Read the original text
        original_text = file.read().decode('utf-8')
        improved_text = improve_agent_details(original_text)

        # Generate agent ID and filenames
        agent_id = get_next_agent_id()
        improved_filename = f"agent_{agent_id:04d}.txt"
        original_filename = f"agent_{agent_id:04d}_original.txt"

        improved_path = os.path.join(UPLOAD_FOLDER, improved_filename)
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)

        # Save improved version
        with open(improved_path, 'w', encoding='utf-8') as f:
            f.write(improved_text)

        # Save original version
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(original_text)

        # Load and update metadata
        metadata = load_metadata()
        agent_metadata = {
            'id': agent_id,
            'filename': improved_filename,
            'original_filename': original_filename,
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(improved_path),
            'status': 'registered'
        }
        metadata[str(agent_id)] = agent_metadata
        save_metadata(metadata)

        return jsonify({
            'success': True,
            'agent_id': agent_id,
            'message': f'Agent registered and improved successfully with ID: {agent_id}',
            'filename': improved_filename,
            'original_filename': original_filename
        }), 201

    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@app.route('/agents', methods=['GET'])
def list_agents():
    """Get list of all registered agents"""
    try:
        metadata = load_metadata()
        return jsonify({
            'agents': metadata,
            'total_count': len(metadata)
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve agents: {str(e)}'}), 500


@app.route('/agent/<int:agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get specific agent information"""
    try:
        metadata = load_metadata()
        agent_id_str = str(agent_id)

        if agent_id_str not in metadata:
            return jsonify({'error': 'Agent not found'}), 404

        return jsonify(metadata[agent_id_str]), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve agent: {str(e)}'}), 500


@app.route('/agent/<int:agent_id>/chat', methods=['POST'])
def chat_with_agent(agent_id):
    """Chat with an agent"""
    question = request.form.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    try:
        metadata = load_metadata()
        agent_id_str = str(agent_id)

        if agent_id_str not in metadata:
            return jsonify({'error': 'Agent not found'}), 404

        agent_info = metadata[agent_id_str]
        agent_filename = agent_info.get('filename')
        if not agent_filename:
            return jsonify({'error': 'Agent file not found in metadata'}), 404

        agent_file_path = os.path.join('data', agent_filename)
        if not os.path.exists(agent_file_path):
            return jsonify({'error': 'Agent file does not exist on disk'}), 404

        with open(agent_file_path, 'r', encoding='utf-8') as f:
            document = f.read()
        answer = generate_answer(question, [document])
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve agent: {str(e)}'}), 500


@app.route('/modelstatus', methods=['GET'])
def model_status():
    from qamodel import model_ready
    if model_ready:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "loading"}), 202




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)