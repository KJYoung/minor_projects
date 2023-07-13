import React, { useState } from 'react';
import { styled } from 'styled-components';
import { Button } from '@mui/material';
import { useDispatch } from 'react-redux';
import { TrxnElement, deleteTrxn, editTrxn } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TypeBubble';
import { GetDateTimeFormatFromDjango } from '../../utils/DateTime';
import { EditAmountInput } from './AmountInput';
import { ViewMode } from '../../containers/TrxnMain';

interface TrxnGridHeaderProps {
    viewMode: ViewMode
};

interface TrxnGridItemProps extends TrxnGridHeaderProps {
    item: TrxnElement,
    isEditing: boolean,
    setEditID: React.Dispatch<React.SetStateAction<number>>
};

interface TrxnGridNavProps {
    viewMode: ViewMode,
    setViewMode: React.Dispatch<React.SetStateAction<ViewMode>>
}
export function TrxnGridNav({viewMode, setViewMode} : TrxnGridNavProps) {
    return <TrxnGridNavWrapper>
        <TrxnGridNavSpan active={(viewMode === ViewMode.Detail).toString()} onClick={() => setViewMode(ViewMode.Detail)}>Detailed</TrxnGridNavSpan>
        <TrxnGridNavSpan active={(viewMode === ViewMode.Graph).toString()} onClick={() => setViewMode(ViewMode.Graph)}>Graphic</TrxnGridNavSpan>
    </TrxnGridNavWrapper>
};

const TrxnGridNavWrapper = styled.div`
    display  : flex;
    width: 100%;
    justify-content: center;
    
    margin-bottom: 15px;
`;
const TrxnGridNavSpan = styled.span<{ active: string }>`
    font-size: 22px;
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    margin-left: 20px;
`;

export function TrxnGridHeader({ viewMode }: TrxnGridHeaderProps ) {
  if(viewMode === ViewMode.Detail){
      return (
        <TrxnGridDetailHeaderDiv>
            <span>ID</span>
            <span>Date</span>
            <span>Period</span>
            <span>Type</span>
            <span>Amount</span>
            <span>Memo</span>
            <span>...</span>
        </TrxnGridDetailHeaderDiv>
      );
  }else if(viewMode === ViewMode.Graph){
      return (
        <TrxnGridGraphicHeaderDiv>
            <span>Date</span>
            <span>Type</span>
            <span>Amount</span>
        </TrxnGridGraphicHeaderDiv>
      );
  }else{
    return <></>
  }
};
export function TrxnGridItem({ item, isEditing, setEditID, viewMode }: TrxnGridItemProps) {
  const [trxnItem, setTrxnItem] = useState<TrxnElement>(item);

  const dispatch = useDispatch<AppDispatch>();

  if(viewMode === ViewMode.Detail){
    return (<TrxnGridDetailItemDiv key={item.id}>
        <span>{item.id}</span>
        <span>{GetDateTimeFormatFromDjango(item.date, true)}</span>
        <span>{item.period > 0 ? item.period : '-'}</span>
        <span>{item.type.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
        
        {/* AMOUNT */}
        {isEditing ? <div>
            <EditAmountInput amount={trxnItem.amount} setAmount={setTrxnItem} />

              {/* <input value={trxnItem.amount} onChange={(e) => setTrxnItem((item) => { return { ...item, amount: Number(e.target.value) } })}/> */}
            </div> 
            : 
            <span>{item.amount}</span>
        }

        {/* MEMO */}
        {isEditing ? <div>
              <input value={trxnItem.memo} onChange={(e) => setTrxnItem((item) => { return { ...item, memo: e.target.value } })}/>
            </div> 
            : 
            <span>{item.memo}</span>
        }
        <div>
            {isEditing && <Button variant={"contained"} disabled={trxnItem.memo === ""} onClick={() => {
                    dispatch(editTrxn(trxnItem));
                    setEditID(-1);
                }}>수정 완료</Button>}
            {!isEditing && <Button onClick={async () => { setEditID(item.id); setTrxnItem(item); }} variant={"contained"}>수정</Button>}
            {!isEditing && <Button onClick={async () => {
                if (window.confirm('정말 기록을 삭제하시겠습니까?')) {
                    dispatch(deleteTrxn(item.id));
                }}} variant={"contained"}>삭제</Button>}
        </div>
    </TrxnGridDetailItemDiv>);
  }else if(viewMode === ViewMode.Graph){
    return (<TrxnGridGraphicItemDiv key={item.id}>
        <span>{GetDateTimeFormatFromDjango(item.date, true)}</span>
        <span>{item.type.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
        <span>{item.amount}</span>
        {/* MEMO */}
        {/* {isEditing ? <div>
              <input value={trxnItem.memo} onChange={(e) => setTrxnItem((item) => { return { ...item, memo: e.target.value } })}/>
            </div> 
            : 
            <span>{item.memo}</span>
        } */}
    </TrxnGridGraphicItemDiv>);
  }else{
    return <></>
  }
};

const TrxnGridDetailTemplate = styled.div`
    display: grid;
    grid-template-columns: 1fr 3fr 2fr 2fr 2fr 5fr 2fr;
    padding-left: 70px;
    padding-right: 70px;   
`;
const TrxnGridDetailHeaderDiv = styled(TrxnGridDetailTemplate)`
    > span {
        /* border: 1px solid red; */
        text-align: center;
        font-size: 22px;
    }
`;
const TrxnGridDetailItemDiv = styled(TrxnGridDetailTemplate)`
    > span {
        /* border: 1px solid red; */
        text-align: center;
        font-size: 22px;
    }
    input {
        width: 100%;
        height: 100%;
    }
`;

const TrxnGridGraphicTemplate = styled.div`
    display: grid;
    grid-template-columns: 2fr 2fr 2fr;
    padding-left: 70px;
    padding-right: 70px;
    width: 40%;
`;
const TrxnGridGraphicHeaderDiv = styled(TrxnGridGraphicTemplate)`
    > span {
        /* border: 1px solid red; */
        text-align: center;
        font-size: 22px;
    }
`;
const TrxnGridGraphicItemDiv = styled(TrxnGridGraphicTemplate)`
    > span {
        /* border: 1px solid red; */
        text-align: center;
        font-size: 22px;
    }
    input {
        width: 100%;
        height: 100%;
    }
`;