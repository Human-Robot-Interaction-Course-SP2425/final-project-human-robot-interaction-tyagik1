import * as Blockly from 'blockly/core';

// Define the set_position block
export const setPositionBlock = {
  type: 'set_position',
  message0: 'Set all positions:',
  message1: 'tower_1: %1',
  args1: [
    {
      type: 'field_number',
      name: 'TOWER1',
      value: 3,
      min: 1,
      max: 5,
      precision: 0.1,
    },
  ],
  message2: 'tower_2: %1',
  args2: [
    {
      type: 'field_number',
      name: 'TOWER2',
      value: 3,
      min: 1,
      max: 5,
      precision: 0.1,
    },
  ],
  message3: 'tower_3: %1',
  args3: [
    {
      type: 'field_number',
      name: 'TOWER3',
      value: 3,
      min: 1,
      max: 5,
      precision: 0.1,
    },
  ],
  message4: 'base: %1',
  args4: [
    {
      type: 'field_number',
      name: 'BASE',
      value: 3,
      min: 1,
      max: 5,
      precision: 0.1,
    },
  ],
  message5: 'ears: %1',
  args5: [
    {
      type: 'field_number',
      name: 'EARS',
      value: 5,
      min: 1,
      max: 5,
      precision: 0.1,
    },
  ],
  previousStatement: 'Position',
  nextStatement: null,
  colour: 90,
  tooltip: 'Set positions for all motors at once (values between 1-5)',
  helpUrl: '',
};

// Create the block definitions
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray([
  setPositionBlock,
]);

