import React from 'react';
import { styled } from 'styled-components';
import { RoundButton } from '../../utils/Button';

interface TypeInputProps {
    // amount: number,
    // setAmount: React.Dispatch<React.SetStateAction<number>>
}

function TypeInput({}: TypeInputProps) {
  return (
    <TypeInputDiv>
        <RoundButton>+</RoundButton>
    </TypeInputDiv>
  );
}

const TypeInputDiv = styled.div`
    background-color: var(--ls-blue);
    border-radius: 5px;
`;

export default TypeInput;
