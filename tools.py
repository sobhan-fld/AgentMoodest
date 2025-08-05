import json
import os

# Configuration for data-management
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'txt'}

# File to store agent metadata
METADATA_FILE = 'data/agents_metadata.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_metadata():
    """Load existing agent metadata"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """Save agent metadata"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_agent_content_by_id(agent_id):
    """Get content from a specific agent by ID"""
    try:
        metadata = load_metadata()
        agent_id_str = str(agent_id)

        if agent_id_str not in metadata:
            return None

        agent_info = metadata[agent_id_str]
        filepath = os.path.join(UPLOAD_FOLDER, agent_info['filename'])

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return content
    except Exception as e:
        print(f"Error loading agent content: {e}")
        return None


def get_all_agents_list():
    """Get list of all available agents for selection"""
    try:
        metadata = load_metadata()
        agents = []

        for agent_id, agent_info in metadata.items():
            agents.append({
                'id': agent_id,
                'name': agent_info.get('original_filename', f'Agent {agent_id}'),
                'filename': agent_info['filename']
            })

        return agents
    except Exception as e:
        print(f"Error loading agents list: {e}")
        return []


def get_next_agent_id():
    """Get the next available agent ID"""
    metadata = load_metadata()
    if not metadata:
        return 1
    return max(int(agent_id) for agent_id in metadata.keys()) + 1