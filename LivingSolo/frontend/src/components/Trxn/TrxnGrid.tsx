import React, { useState } from 'react';
import { styled } from 'styled-components';
import { Button, TextField } from '@mui/material';
import { useDispatch } from 'react-redux';
import { TrxnElement, deleteTrxn, editTrxn } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TypeBubble';
import { CalMonth, GetDateTimeFormatFromDjango } from '../../utils/DateTime';
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
    const CUR_MONTH = {
        year: (new Date()).getFullYear(),
        month: (new Date()).getMonth() + 1
    };
    const CUR_YEAR = {
        year: (new Date()).getFullYear()
    };
    const [curMonth, setCurMonth] = useState<CalMonth>(CUR_MONTH);
    const [monthMode, setMonthMode] = useState<boolean>(true);
    const [search, setSearch] = useState<string>("");
    
    const TrxnGridNavCalendar = () => {
        const monthHandler = (am: number) => {
            const afterMonth = (curMonth.month !== undefined ? curMonth.month : 0) + am;
            if(afterMonth > 12)
                setCurMonth({year: curMonth.year + Math.floor(afterMonth / 12), month: afterMonth % 12});
            else if(afterMonth <= 0)
                setCurMonth({year: curMonth.year - Math.floor(afterMonth / 12) - 1, month: 12 - ((-afterMonth) % 12)});
            else
                setCurMonth({...curMonth, month: afterMonth});
        };
        const yearHandler = (am: number) => {
            setCurMonth({year: curMonth.year + am});
        };
        const monthModeToggler = () => {
            if(monthMode){ // month => year
                setMonthMode(false);
                setCurMonth({year: curMonth.year, month: undefined});
            }else{
                setMonthMode(true);
                setCurMonth({year: curMonth.year, month: 1});
            }
        }
        return <MonthNavWrapper>
            <MonthNavBtn onClick={() => monthMode ? monthHandler(-1) : yearHandler(-1)}>{"<"}</MonthNavBtn>
            <MonthIndSpan>{`${curMonth.year}년`}{monthMode && ` ${curMonth.month}월`}</MonthIndSpan>
            <MonthNavBtn onClick={() => monthMode ? monthHandler(+1) : yearHandler(+1)}>{">"}</MonthNavBtn>
            <div>
                <button onClick={() => monthModeToggler()}>{monthMode ? "년 단위로" : "월 단위로"}</button>
                <button onClick={() => setCurMonth(monthMode ? CUR_MONTH : CUR_YEAR)}>오늘로</button>
            </div>
        </MonthNavWrapper>
    }

    return <TrxnGridNavWrapper>
        <div>
            <TrxnGridModeBtn active={(viewMode === ViewMode.Detail).toString()} onClick={() => setViewMode(ViewMode.Detail)}>Detailed</TrxnGridModeBtn>
            <TrxnGridModeBtn active={(viewMode === ViewMode.Graph).toString()} onClick={() => setViewMode(ViewMode.Graph)}>Graphic</TrxnGridModeBtn>
        </div>
        {viewMode === ViewMode.Detail && <TrxnGridDetailedSubNav>
            <TrxnGridNavCalendar />
            <div>
                <TextField className="TextField" label="검색" variant="outlined" value={search} onChange={(e) => setSearch(e.target.value)}/>
                <button>검색!</button>
            </div>
        </TrxnGridDetailedSubNav>}
    </TrxnGridNavWrapper>
};

const MonthNavBtn = styled.button`
    background-color: transparent;
    border: none;
    font-size: 25px;
    color: var(--ls-gray_google2);
    cursor: pointer;
    &:hover {
    background-color: var(--ls-gray_google1);
    }
    border-radius: 50%;
    width: 38px;
    height: 38px;
    margin-left: 6px;
    margin-right: 6px;
`;
const MonthIndSpan = styled.div`
    font-size: 24px;
    color: var(--ls-gray_google2);
    width: 140px;
    text-align: center;
`;
const MonthNavWrapper = styled.div`
    width: 100%;
    display: flex;
    flex-direction: row;
    align-items: center;
`;

const TrxnGridNavWrapper = styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
    justify-content: center;
    align-items: center;
    margin-bottom: 15px;
`;
const TrxnGridDetailedSubNav = styled.div`
    width: 100%;
    display: grid;
    grid-template-columns: 2fr 8fr;
    padding-left: 70px;
    padding-right: 70px;

    .TextField {
        width   : 80%;
    }

`;
const TrxnGridModeBtn = styled.span<{ active: string }>`
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
            <div></div>
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
            <span className='amount'>{item.amount.toLocaleString()}</span>
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
        border-right: 1px solid gray;
        text-align: center;
        font-size: 22px;
    }
`;
const TrxnGridDetailItemDiv = styled(TrxnGridDetailTemplate)`
    > span {
        border-right: 1px solid gray;
        text-align: center;
        font-size: 22px;
    }
    .amount {
        padding-right: 10px;
        text-align: right;
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
    .amount {
        text-align: right;
    }
    input {
        width: 100%;
        height: 100%;
    }
`;