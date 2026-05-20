/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {Order} from 'blockly/javascript';
import * as Blockly from 'blockly/core';

// Export all the code generators for our custom blocks,
// but don't register them with Blockly yet.
// This file has no side effects!
export const forBlock = Object.create(null);


forBlock['create_sequence'] = function (
  block: Blockly.Block,
  generator: Blockly.CodeGenerator,
) {
  const name = block.getFieldValue('NAME');
  const frames = generator.statementToCode(block, 'FRAMES');
  
  const code = `window.animationData = {
  "animation": "${name}",
  "frame_list": [${frames}]
};\n`;
  return code;
};

forBlock['create_frame'] = function (
  block: Blockly.Block,
  generator: Blockly.CodeGenerator,
) {
  const millis = block.getFieldValue('MILLIS');
  const positions = generator.statementToCode(block, 'POSITIONS');
  
  // Check if there's a next statement (more frames after this one)
  const hasNext = block.getNextBlock();
  const comma = hasNext ? ',' : '';
  
  const code = `{
    "positions": [${positions}],
    "millis": ${millis}
  }${comma}`;
  return code;
};

forBlock['set_position'] = function (
  block: Blockly.Block,
  generator: Blockly.CodeGenerator,
) {
  const tower1 = block.getFieldValue('TOWER1');
  const tower2 = block.getFieldValue('TOWER2');
  const tower3 = block.getFieldValue('TOWER3');
  const base = block.getFieldValue('BASE');
  const ears = block.getFieldValue('EARS');

  const code = `{
      "dof": "tower_1",
      "pos": ${tower1}
    },
    {
      "dof": "tower_2",
      "pos": ${tower2}
    },
    {
      "dof": "tower_3",
      "pos": ${tower3}
    },
    {
      "dof": "base",
      "pos": ${base}
    },
    {
      "dof": "ears",
      "pos": ${ears}
    }`;
  return code;
};
