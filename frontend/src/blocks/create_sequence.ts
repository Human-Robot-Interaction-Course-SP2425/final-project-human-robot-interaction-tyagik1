import * as Blockly from 'blockly/core';

// Define the create_sequence block
export const createSequenceBlock = {
  type: 'create_sequence',
  message0: 'Create sequence named %1',
  args0: [
    {
      type: 'field_input',
      name: 'NAME',
      text: 'my_animation',
    },
  ],
  message1: 'frames %1',
  args1: [
    {
      type: 'input_statement',
      name: 'FRAMES',
      check: 'Frame',
    },
  ],
  colour: 230,
  tooltip: 'Create a new animation sequence',
  helpUrl: '',
};

// Create the block definitions
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray([
  createSequenceBlock,
]);

