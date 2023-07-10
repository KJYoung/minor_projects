import React, { useState } from 'react';
import { styled } from 'styled-components';
import { EditTextInput } from '../../utils/EditText';
import { RoundButton } from '../../utils/Button';
import { TrxnElement } from '../../store/slices/trxn';

interface NewAmountInputProps {
    amount: number,
    setAmount: React.Dispatch<React.SetStateAction<number>>
}
interface EditAmountInputProps {
    amount: number,
    setAmount: React.Dispatch<React.SetStateAction<TrxnElement>>
}

export function NewAmountInput({amount, setAmount}: NewAmountInputProps) {
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
            <AmountShortCutWrapper>
                <RoundButton onClick={() => setExtended((e) => !e)}>{extended ? '-' : '+'}</RoundButton>
                {extended && <AmountShortCutDiv>
                    <button onClick={() => addAmount(1000)}>+1000</button>
                    <button onClick={() => addAmount(-1000)} disabled={amount <= 0}>-1000</button>
                    <button onClick={() => addAmount(+10000)}>+10000</button>
                    <button onClick={() => addAmount(-10000)} disabled={amount <= 0}>-10000</button>
                    <button onClick={() => addAmount(+5000)}>+5000</button>
                    <button onClick={() => addAmount(-5000)} disabled={amount <= 0}>-5000</button>
                    <button onClick={() => setAmount(0)}>Clear</button>
                </AmountShortCutDiv>}
            </AmountShortCutWrapper>
        </div>
    </AmountInputDiv>
  );
}

export function EditAmountInput({amount, setAmount}: EditAmountInputProps) {
    const [extended, setExtended] = useState<boolean>(false);
    const addAmount = (param1: number) => { // Threshold 0.
      setAmount((am) => {
        return {...am, amount: (am.amount + param1 >= 0) ? (am.amount + param1) : 0};
      });
    };
    const setAmount_ = (value: number) => {
      setAmount((am) => {
        return {...am, amount: (value >= 0) ? value : 0};
      });
    }
  
    return (
      <AmountInputDiv>
          <div>
              <AmountEditText placeholder='금액' type='number' value={amount.toString()} onChange={(e) => {
              try{
                  const num = Number(e.target.value)
                  setAmount_(num >= 0 ? num : 0);
              } catch {
                  console.log("NaN" + e.target.value);
              };  
              }} pattern="[0-9]+" min={0}/>
              <AmountShortCutWrapper>
                  <RoundButton onClick={() => setExtended((e) => !e)}>{extended ? '-' : '+'}</RoundButton>
                  {extended && <AmountShortCutDiv>
                      <button onClick={() => addAmount(1000)}>+1000</button>
                      <button onClick={() => addAmount(-1000)} disabled={amount <= 0}>-1000</button>
                      <button onClick={() => addAmount(+10000)}>+10000</button>
                      <button onClick={() => addAmount(-10000)} disabled={amount <= 0}>-10000</button>
                      <button onClick={() => addAmount(+5000)}>+5000</button>
                      <button onClick={() => addAmount(-5000)} disabled={amount <= 0}>-5000</button>
                      <button onClick={() => setAmount_(0)}>Clear</button>
                  </AmountShortCutDiv>}
              </AmountShortCutWrapper>
          </div>
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
const AmountShortCutWrapper = styled.div`
    position: relative;

    z-index: 2;
`;
const AmountShortCutDiv = styled.div`
    background-color: var(--ls-white);
    display: grid;
    grid-template-columns: 2fr 2fr;

    width: 200px;
    height: 300px;
    position: absolute;
    top: 35px;
    left: -100px;

    button {
        margin: 5px;
        /* background-color: aliceblue; */
    }
`;
