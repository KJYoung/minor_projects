import React, { useState } from 'react';
import { styled } from 'styled-components';
import { EditTextInput } from '../../utils/EditText';
import { RoundButton } from '../../utils/Button';

interface AmountInputProps {
    amount: number,
    setAmount: React.Dispatch<React.SetStateAction<number>>
}

function AmountInput({amount, setAmount}: AmountInputProps) {
  const [extended, setExtended] = useState<boolean>(false);
  const addAmount = (param1: number) => { // Threshold 0.
    setAmount((am) => (am + param1 >= 0) ? (am + param1) : 0);
  };

  return (
    <AmountInputDiv>
        <div>
            <AmountEditText placeholder='금액' type='number' value={amount.toString()} onChange={(e) => {
            try{
                const num = Number(e.target.value)
                setAmount(num >= 0 ? num : 0);
            } catch {
                console.log("NaN" + e.target.value);
            };  
            }} pattern="[0-9]+" min={0}/>
            <RoundButton onClick={() => setExtended((e) => !e)}>{extended ? '-' : '+'}</RoundButton>
        </div>
        {extended && <div>
            <button onClick={() => addAmount(1000)}>+1000</button>
            <button onClick={() => addAmount(-1000)} disabled={amount <= 0}>-1000</button>
            <button onClick={() => addAmount(+10000)}>+10000</button>
            <button onClick={() => addAmount(-10000)} disabled={amount <= 0}>-10000</button>
            <button onClick={() => addAmount(+5000)}>+5000</button>
            <button onClick={() => addAmount(-5000)} disabled={amount <= 0}>-5000</button>
            <button onClick={() => setAmount(0)}>Clear</button>
        </div>}
    </AmountInputDiv>
  );
}

const AmountInputDiv = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    width: 100%;

    > div {
        display: flex;
        align-items: center;
        height: fit-content;
    }
`;

const AmountEditText = styled(EditTextInput)`
    margin-right: 10px;
    height: 35px;
`;
export default AmountInput;
