/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/*
This toolbox contains nearly every single built-in block that Blockly offers,
in addition to the custom block 'add_text' this sample app adds.
You probably don't need every single block, and should consider either rewriting
your toolbox from scratch, or carefully choosing whether you need each block
listed here.
*/

export const toolbox = {
  kind: 'categoryToolbox',
  contents: [
    {
      kind: 'category',
      name: 'Gestures',
      colour: 120,
      contents: [
        {
          kind: 'block',
          type: 'create_sequence',
        },
        {
          kind: 'block',
          type: 'create_frame',
        },
        {
          kind: 'block',
          type: 'set_position',
        },
      ],
    },
  ],
};
