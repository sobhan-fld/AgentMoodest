from flask import Flask, render_template, request, redirect, jsonify
from tools import *
from datetime import datetime
app = Flask(__name__)

@app.route('/agentregister', methods=['POST'])
def agent_register():
    """
    Endpoint to register an agent by uploading a text file
    Returns: JSON with agent_id and status
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .txt files are allowed'}), 400

        # Get next agent ID
        agent_id = get_next_agent_id()

        # Create filename with agent ID
        filename = f"agent_{agent_id:04d}.txt"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save the file
        file.save(filepath)

        # Load existing metadata
        metadata = load_metadata()

        # Create agent metadata
        agent_metadata = {
            'id': agent_id,
            'filename': filename,
            'original_filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(filepath),
            'status': 'registered'
        }

        # Add to metadata
        metadata[str(agent_id)] = agent_metadata

        # Save metadata
        save_metadata(metadata)

        return jsonify({
            'success': True,
            'agent_id': agent_id,
            'message': f'Agent registered successfully with ID: {agent_id}',
            'filename': filename
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



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)