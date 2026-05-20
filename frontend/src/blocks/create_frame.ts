import * as Blockly from 'blockly/core';

// Define the create_frame block
export const createFrameBlock = {
  type: 'create_frame',
  message0: 'Frame at %1 milliseconds',
  args0: [
    {
      type: 'field_number',
      name: 'MILLIS',
      value: 0,
      min: 0,
    },
  ],
  message1: 'positions %1',
  args1: [
    {
      type: 'input_statement',
      name: 'POSITIONS',
      check: 'Position',
    },
  ],
  previousStatement: 'Frame',
  nextStatement: 'Frame',
  colour: 160,
  tooltip: 'Create a frame with timestamp',
  helpUrl: '',
};

// Create the block definitions
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray([
  createFrameBlock,
]);

