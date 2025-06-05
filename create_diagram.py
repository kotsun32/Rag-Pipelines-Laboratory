import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
colors = {
    'interface': '#E3F2FD',
    'framework': '#FFF3E0', 
    'vector': '#E8F5E8',
    'retrieval': '#FCE4EC',
    'metrics': '#F3E5F5',
    'data': '#FFEBEE'
}

# Title
ax.text(7, 9.5, 'RAG Pipeline Laboratory Architecture', 
        fontsize=20, fontweight='bold', ha='center')

# User Interface Layer
interface_box = FancyBboxPatch((0.5, 8), 13, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['interface'],
                              edgecolor='#1976D2',
                              linewidth=2)
ax.add_patch(interface_box)
ax.text(7, 8.5, 'Gradio Web Interface', fontsize=14, fontweight='bold', ha='center')
ax.text(2, 8.2, '• Pipeline Setup', fontsize=10, ha='left')
ax.text(5.5, 8.2, '• Query Interface', fontsize=10, ha='left')
ax.text(8.5, 8.2, '• Comparison Dashboard', fontsize=10, ha='left')
ax.text(12, 8.2, '• Analytics', fontsize=10, ha='left')

# Framework Layer
framework_boxes = [
    {'name': 'Custom\nPipeline', 'x': 1, 'color': colors['framework']},
    {'name': 'LangChain', 'x': 4, 'color': colors['framework']},
    {'name': 'LlamaIndex', 'x': 7, 'color': colors['framework']},
    {'name': 'LangGraph\n(Coming Soon)', 'x': 10, 'color': colors['framework']}
]

for framework in framework_boxes:
    box = FancyBboxPatch((framework['x']-0.7, 6.5), 2.4, 1,
                        boxstyle="round,pad=0.1",
                        facecolor=framework['color'],
                        edgecolor='#F57C00',
                        linewidth=1.5)
    ax.add_patch(box)
    ax.text(framework['x']+0.5, 7, framework['name'], 
           fontsize=10, fontweight='bold', ha='center', va='center')

ax.text(0.5, 7.2, 'Frameworks:', fontsize=12, fontweight='bold')

# Vector Store Layer
vector_boxes = [
    {'name': 'FAISS', 'x': 1.5},
    {'name': 'Chroma', 'x': 4.5},
    {'name': 'Weaviate', 'x': 7.5},
    {'name': 'Pinecone', 'x': 10.5}
]

for vector in vector_boxes:
    box = FancyBboxPatch((vector['x']-0.6, 5), 1.2, 0.8,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['vector'],
                        edgecolor='#388E3C',
                        linewidth=1.5)
    ax.add_patch(box)
    ax.text(vector['x'], 5.4, vector['name'], 
           fontsize=9, fontweight='bold', ha='center', va='center')

ax.text(0.5, 5.6, 'Vector Stores:', fontsize=12, fontweight='bold')

# Retrieval Methods Layer
retrieval_boxes = [
    {'name': 'Semantic\nSearch', 'x': 2},
    {'name': 'Keyword\nSearch', 'x': 5},
    {'name': 'Hybrid\nRetrieval', 'x': 8},
    {'name': 'BM25\nRanking', 'x': 11}
]

for retrieval in retrieval_boxes:
    box = FancyBboxPatch((retrieval['x']-0.7, 3.5), 1.4, 0.8,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['retrieval'],
                        edgecolor='#C2185B',
                        linewidth=1.5)
    ax.add_patch(box)
    ax.text(retrieval['x'], 3.9, retrieval['name'], 
           fontsize=9, fontweight='bold', ha='center', va='center')

ax.text(0.5, 4.1, 'Retrieval Methods:', fontsize=12, fontweight='bold')

# Performance Metrics Layer
metrics_box = FancyBboxPatch((2, 2), 10, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['metrics'],
                           edgecolor='#7B1FA2',
                           linewidth=1.5)
ax.add_patch(metrics_box)
ax.text(2.5, 2.6, 'Performance Metrics:', fontsize=12, fontweight='bold')
ax.text(3, 2.2, '• Retrieval Time', fontsize=10)
ax.text(5.5, 2.2, '• Generation Time', fontsize=10)
ax.text(8, 2.2, '• Total Response Time', fontsize=10)
ax.text(10.5, 2.2, '• Accuracy Metrics', fontsize=10)

# Data Flow Layer
data_box = FancyBboxPatch((1, 0.5), 12, 0.8,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['data'],
                         edgecolor='#D32F2F',
                         linewidth=1.5)
ax.add_patch(data_box)
ax.text(1.5, 1.1, 'Data Flow:', fontsize=12, fontweight='bold')
ax.text(3, 0.7, 'Documents → Chunking → Embedding → Vector Store → Retrieval → LLM → Response', 
        fontsize=10, ha='left')

# Add arrows showing data flow
arrow_props = dict(arrowstyle='->', lw=2, color='#424242')

# Interface to Frameworks
ax.annotate('', xy=(7, 7.5), xytext=(7, 8),
           arrowprops=arrow_props)

# Frameworks to Vector Stores
for i, framework in enumerate(framework_boxes):
    if i < len(vector_boxes):
        ax.annotate('', xy=(vector_boxes[i]['x'], 5.8), 
                   xytext=(framework['x']+0.5, 6.5),
                   arrowprops=arrow_props)

# Vector Stores to Retrieval Methods
for i, vector in enumerate(vector_boxes):
    if i < len(retrieval_boxes):
        ax.annotate('', xy=(retrieval_boxes[i]['x'], 4.3), 
                   xytext=(vector['x'], 5),
                   arrowprops=arrow_props)

# Retrieval to Metrics
ax.annotate('', xy=(7, 2.8), xytext=(7, 3.5),
           arrowprops=arrow_props)

# Add configuration flow indicators
ax.text(13.2, 6, 'User\nConfig', fontsize=10, ha='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFECB3'))

# Add legend
legend_elements = [
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['interface'], label='User Interface'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['framework'], label='Frameworks'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['vector'], label='Vector Stores'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['retrieval'], label='Retrieval Methods'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['metrics'], label='Performance Metrics'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['data'], label='Data Processing')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('/home/sunny/Documents/Rag_pipelines/architecture_diagram.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Create a simpler flow diagram
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(6, 7.5, 'RAG Pipeline Flow Diagram', 
        fontsize=18, fontweight='bold', ha='center')

# Flow steps
steps = [
    {'name': 'User Query', 'x': 1, 'y': 6, 'color': '#E3F2FD'},
    {'name': 'Document\nEmbedding', 'x': 3.5, 'y': 6, 'color': '#FFF3E0'},
    {'name': 'Vector\nSearch', 'x': 6, 'y': 6, 'color': '#E8F5E8'},
    {'name': 'Context\nRetrieval', 'x': 8.5, 'y': 6, 'color': '#FCE4EC'},
    {'name': 'LLM\nGeneration', 'x': 11, 'y': 6, 'color': '#F3E5F5'}
]

# Configuration options
config_steps = [
    {'name': 'Framework\nSelection', 'x': 2, 'y': 4, 'color': '#FFEBEE'},
    {'name': 'Vector Store\nChoice', 'x': 5, 'y': 4, 'color': '#FFEBEE'},
    {'name': 'Retrieval\nMethod', 'x': 8, 'y': 4, 'color': '#FFEBEE'},
    {'name': 'Model\nParameters', 'x': 11, 'y': 4, 'color': '#FFEBEE'}
]

# Draw main flow
for i, step in enumerate(steps):
    box = FancyBboxPatch((step['x']-0.7, step['y']-0.4), 1.4, 0.8,
                        boxstyle="round,pad=0.1",
                        facecolor=step['color'],
                        edgecolor='#424242',
                        linewidth=2)
    ax.add_patch(box)
    ax.text(step['x'], step['y'], step['name'], 
           fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Add arrows between steps
    if i < len(steps) - 1:
        ax.annotate('', xy=(steps[i+1]['x']-0.7, steps[i+1]['y']), 
                   xytext=(step['x']+0.7, step['y']),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#1976D2'))

# Draw configuration options
for step in config_steps:
    box = FancyBboxPatch((step['x']-0.7, step['y']-0.4), 1.4, 0.8,
                        boxstyle="round,pad=0.1",
                        facecolor=step['color'],
                        edgecolor='#D32F2F',
                        linewidth=1.5,
                        linestyle='--')
    ax.add_patch(box)
    ax.text(step['x'], step['y'], step['name'], 
           fontsize=9, ha='center', va='center')

# Add configuration arrows
for i, config in enumerate(config_steps):
    main_step = steps[i+1] if i+1 < len(steps) else steps[-1]
    ax.annotate('', xy=(main_step['x'], main_step['y']-0.4), 
               xytext=(config['x'], config['y']+0.4),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#D32F2F', 
                             linestyle='--', alpha=0.7))

# Add metrics box
metrics_box = FancyBboxPatch((4, 2), 4, 1,
                           boxstyle="round,pad=0.1",
                           facecolor='#F3E5F5',
                           edgecolor='#7B1FA2',
                           linewidth=2)
ax.add_patch(metrics_box)
ax.text(6, 2.7, 'Performance Metrics', fontsize=12, fontweight='bold', ha='center')
ax.text(6, 2.3, 'Retrieval Time • Generation Time • Accuracy', 
        fontsize=10, ha='center')

# Add feedback arrow
ax.annotate('', xy=(6, 5.6), xytext=(6, 3),
           arrowprops=dict(arrowstyle='<->', lw=2, color='#7B1FA2'))

# Add labels
ax.text(0.5, 6.5, 'Main Flow:', fontsize=12, fontweight='bold', color='#1976D2')
ax.text(0.5, 4.5, 'Configuration:', fontsize=12, fontweight='bold', color='#D32F2F')
ax.text(0.5, 2.5, 'Feedback:', fontsize=12, fontweight='bold', color='#7B1FA2')

plt.tight_layout()
plt.savefig('/home/sunny/Documents/Rag_pipelines/flow_diagram.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Diagrams created successfully!")
print("📁 Files generated:")
print("   - architecture_diagram.png")
print("   - flow_diagram.png")