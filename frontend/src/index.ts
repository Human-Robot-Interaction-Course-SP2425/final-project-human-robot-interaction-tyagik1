/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as Blockly from 'blockly';
import {blocks as setPositionBlocks} from './blocks/set_position';
import {blocks as createSequenceBlocks} from './blocks/create_sequence';
import {blocks as createFrameBlocks} from './blocks/create_frame';
import {forBlock} from './generators/javascript';
import {javascriptGenerator} from 'blockly/javascript';
import {save, load} from './serialization';
import {toolbox} from './toolbox';
import './index.css';

// Register the blocks and generator with Blockly
Blockly.common.defineBlocks(setPositionBlocks);
Blockly.common.defineBlocks(createSequenceBlocks);
Blockly.common.defineBlocks(createFrameBlocks);
Object.assign(javascriptGenerator.forBlock, forBlock);

// Set up UI elements and inject Blockly
const codeDiv = document.getElementById('generatedCode')?.querySelector('code');
const blocklyDiv = document.getElementById('blocklyDiv');

// Global state to store animation data
let animationData: any = null;

// Make animationData accessible globally for the generated code
(window as any).animationData = animationData;

// API base URL - change this to your FastAPI server URL
const API_BASE_URL = 'http://localhost:8000';

// Cache for the full list of sequence names (used for filtering)
let allSequenceNames: string[] = [];

// --- API Functions ---

async function resetRobot() {
  try {
    const response = await fetch(`${API_BASE_URL}/reset`, {
      method: 'POST',
    });
    if (response.ok) {
      alert('Robot reset successfully!');
    } else {
      alert('Failed to reset robot');
    }
  } catch (error) {
    console.error('Error resetting robot:', error);
    alert('Error connecting to robot. Make sure the server is running.');
  }
}

async function playSequence() {
  if (!animationData) {
    alert('No sequence to play. Create a sequence using the blocks first.');
    return;
  }

  try {
    const payload = {
      animation: animationData.animation,
      frame_list: animationData.frame_list,
    };

    const response = await fetch(`${API_BASE_URL}/sequences/play`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const result = await response.json();
      const mode = result.simulation ? ' (simulation mode)' : '';
      alert(
        `Sequence '${result.animation}' completed successfully${mode}!\n` +
          `Frames: ${result.frame_count}`,
      );
    } else {
      const error = await response.json();
      console.error('API Error:', error);
      alert(`Failed to play sequence: ${error.detail || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Error playing sequence:', error);
    alert(
      'Error connecting to robot. Make sure the server is running at ' +
        API_BASE_URL,
    );
  }
}

async function saveSequence() {
  if (!animationData) {
    alert('No sequence to save. Create a sequence using the blocks first.');
    return;
  }

  try {
    const payload = {
      animation: animationData.animation,
      frame_list: animationData.frame_list,
    };

    const response = await fetch(`${API_BASE_URL}/sequences`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const result = await response.json();
      alert(
        `Sequence '${result.animation}' saved successfully!\n` +
          `File: ${result.file}`,
      );
      loadSequenceNames();
    } else {
      const error = await response.json();
      console.error('API Error:', error);
      alert(`Failed to save sequence: ${error.detail || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Error saving sequence:', error);
    alert(
      'Error connecting to server. Make sure the server is running at ' +
        API_BASE_URL,
    );
  }
}

async function loadSequenceNames() {
  const listEl = document.getElementById('sequenceList');
  if (!listEl) return;

  try {
    const response = await fetch(`${API_BASE_URL}/sequences/names`);
    if (!response.ok) {
      listEl.innerHTML =
        '<li class="empty-msg">Failed to load sequences</li>';
      return;
    }

    const data = await response.json();
    allSequenceNames = data.names || [];

    renderSequenceList(allSequenceNames);
  } catch (error) {
    console.error('Error loading sequence names:', error);
    listEl.innerHTML =
      '<li class="empty-msg">Server unavailable</li>';
    allSequenceNames = [];
  }
}

function renderSequenceList(names: string[]) {
  const listEl = document.getElementById('sequenceList');
  if (!listEl) return;

  if (names.length === 0) {
    listEl.innerHTML = '<li class="empty-msg">No sequences found</li>';
    return;
  }

  listEl.innerHTML = '';
  for (const name of names) {
    const li = document.createElement('li');

    const nameSpan = document.createElement('span');
    nameSpan.className = 'seq-name';
    nameSpan.textContent = name;
    nameSpan.title = `Click to preview "${name}"`;
    nameSpan.addEventListener('click', () => previewSavedSequence(name));

    const playBtn = document.createElement('button');
    playBtn.className = 'seq-play-btn';
    playBtn.textContent = 'Play';
    playBtn.addEventListener('click', () => playSavedSequence(name));

    li.appendChild(nameSpan);
    li.appendChild(playBtn);
    listEl.appendChild(li);
  }
}

function filterSequenceList(query: string) {
  const lower = query.toLowerCase().trim();
  if (!lower) {
    renderSequenceList(allSequenceNames);
    return;
  }
  const filtered = allSequenceNames.filter((n) =>
    n.toLowerCase().includes(lower),
  );
  renderSequenceList(filtered);
}

async function previewSavedSequence(name: string) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/sequences/${encodeURIComponent(name)}`,
    );
    if (!response.ok) {
      alert(`Sequence '${name}' not found.`);
      return;
    }
    const data = await response.json();
    if (codeDiv) {
      codeDiv.textContent = JSON.stringify(data, null, 2);
    }
  } catch (error) {
    console.error('Error loading sequence:', error);
    alert('Error connecting to server.');
  }
}

async function playSavedSequence(name: string) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/sequences/${encodeURIComponent(name)}/play`,
    );
    if (response.ok) {
      const result = await response.json();
      const mode = result.simulation ? ' (simulation)' : '';
      alert(`Sequence '${name}' completed${mode}!`);
    } else {
      const error = await response.json();
      alert(
        `Failed to play sequence: ${error.detail || 'Unknown error'}`,
      );
    }
  } catch (error) {
    console.error('Error playing saved sequence:', error);
    alert('Error connecting to server.');
  }
}

// --- Blockly Workspace Setup ---

if (!blocklyDiv) {
  throw new Error(`div with id 'blocklyDiv' not found`);
}
const ws = Blockly.inject(blocklyDiv, {toolbox});

const runCode = () => {
  const code = javascriptGenerator.workspaceToCode(ws as Blockly.Workspace);

  if (code.trim()) {
    try {
      animationData = null;
      (window as any).animationData = null;

      eval(code);

      animationData = (window as any).animationData;

      if (animationData && codeDiv) {
        codeDiv.textContent = JSON.stringify(animationData, null, 2);
      } else if (codeDiv) {
        codeDiv.textContent = code;
      }
    } catch (e) {
      console.error('Error executing generated code:', e);
      if (codeDiv) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        codeDiv.textContent = `Error: ${errorMessage}`;
      }
    }
  } else {
    animationData = null;
    (window as any).animationData = null;
    if (codeDiv) {
      codeDiv.textContent = '';
    }
  }
};

if (ws) {
  load(ws);
  runCode();

  ws.addChangeListener((e: Blockly.Events.Abstract) => {
    if (e.isUiEvent) return;
    save(ws);
  });

  ws.addChangeListener((e: Blockly.Events.Abstract) => {
    if (
      e.isUiEvent ||
      e.type == Blockly.Events.FINISHED_LOADING ||
      ws.isDragging()
    ) {
      return;
    }
    runCode();
  });
}

// --- Button Event Listeners ---

const resetBtn = document.getElementById('resetBtn');
const playBtn = document.getElementById('playBtn');
const saveBtn = document.getElementById('saveBtn');
const refreshBtn = document.getElementById('refreshBtn');
const searchInput = document.getElementById(
  'sequenceSearch',
) as HTMLInputElement | null;

if (resetBtn) {
  resetBtn.addEventListener('click', resetRobot);
}

if (playBtn) {
  playBtn.addEventListener('click', playSequence);
}

if (saveBtn) {
  saveBtn.addEventListener('click', saveSequence);
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', loadSequenceNames);
}

if (searchInput) {
  searchInput.addEventListener('input', () => {
    filterSequenceList(searchInput.value);
  });
}

// Load saved sequence names on startup
loadSequenceNames();
