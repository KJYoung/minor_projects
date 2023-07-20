import React, { useState } from 'react';
import { styled } from 'styled-components';
import { Button, TextField } from '@mui/material';
import { useDispatch } from 'react-redux';
import { TrxnElement, TrxnFetchReqType, deleteTrxn, editTrxn, fetchTrxns } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TagBubble';
import { CUR_MONTH, CUR_YEAR, CalMonth, GetDateTimeFormatFromDjango } from '../../utils/DateTime';
import { EditAmountInput } from './AmountInput';
import { ViewMode } from '../../containers/TrxnMain';
import { RoundButton } from '../../utils/Button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPencil, faX } from '@fortawesome/free-solid-svg-icons';

interface TrxnGridHeaderProps {
    viewMode: ViewMode
};

interface TrxnGridItemProps extends TrxnGridHeaderProps {
    index: number,
    item: TrxnElement,
    isEditing: boolean,
    setEditID: React.Dispatch<React.SetStateAction<number>>
};

interface TrxnGridNavProps {
    viewMode: ViewMode,
    setViewMode: React.Dispatch<React.SetStateAction<ViewMode>>
}

export function TrxnGridNav({viewMode, setViewMode} : TrxnGridNavProps) {
    const [curMonth, setCurMonth] = useState<CalMonth>(CUR_MONTH);
    const [monthMode, setMonthMode] = useState<boolean>(true);
    const [search, setSearch] = useState<string>("");
    const fetchObj: TrxnFetchReqType = {
        yearMonth: CUR_MONTH,
        searchKeyword: "",
        dayCombined: false
    };

    const dispatch = useDispatch<AppDispatch>();
    const TrxnGridNavCalendar = () => {
        const monthHandler = (am: number) => {
            const afterMonth = (curMonth.month !== undefined ? curMonth.month : 0) + am;

            if(afterMonth > 12)
                fetchObj.yearMonth = {year: curMonth.year + Math.floor(afterMonth / 12), month: afterMonth % 12};
            else if(afterMonth <= 0)
                fetchObj.yearMonth = {year: curMonth.year - Math.floor(afterMonth / 12) - 1, month: 12 - ((-afterMonth) % 12)};
            else
                fetchObj.yearMonth = {...curMonth, month: afterMonth};
            
            dispatch(fetchTrxns(fetchObj));
            setCurMonth(fetchObj.yearMonth);
        };
        const yearHandler = (am: number) => {
            fetchObj.yearMonth = {year: curMonth.year + am};
            setCurMonth(fetchObj.yearMonth);
            dispatch(fetchTrxns(fetchObj));
        };
        const todayHandler = () => {
            fetchObj.yearMonth = monthMode ? CUR_MONTH : CUR_YEAR;
            setCurMonth(fetchObj.yearMonth);
            dispatch(fetchTrxns(fetchObj));
        };
        const monthModeToggler = () => {
            if(monthMode){ // month => year
                fetchObj.yearMonth = {year: curMonth.year, month: undefined};
                setMonthMode(false);
            }else{
                fetchObj.yearMonth = {year: curMonth.year, month: 1};
                setMonthMode(true);
            }
            setCurMonth(fetchObj.yearMonth);
            dispatch(fetchTrxns(fetchObj));
        }
        return <MonthNavWrapper>
            <MonthNavBtn onClick={() => monthMode ? monthHandler(-1) : yearHandler(-1)}>{"<"}</MonthNavBtn>
            <MonthIndSpan>{`${curMonth.year}년`}{monthMode && ` ${curMonth.month}월`}</MonthIndSpan>
            <MonthNavBtn onClick={() => monthMode ? monthHandler(+1) : yearHandler(+1)}>{">"}</MonthNavBtn>
            <div>
                <MonthNavSubBtn onClick={() => monthModeToggler()}>{monthMode ? "년 단위로" : "월 단위로"}</MonthNavSubBtn>
                <MonthNavSubBtn onClick={() => todayHandler()}>오늘로</MonthNavSubBtn>
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
            <SearchNavWrapper>
                <TextField className="TextField" label="검색" variant="outlined" value={search} onChange={(e) => setSearch(e.target.value)}/>
                <div>
                    <MonthNavSubBtn onClick={() => {
                        fetchObj.searchKeyword = search;
                        dispatch(fetchTrxns(fetchObj));
                    }} disabled={search === ""}>검색!</MonthNavSubBtn>
                    <MonthNavSubBtn onClick={() => {
                        setSearch("");
                        fetchObj.searchKeyword = "";
                        dispatch(fetchTrxns(fetchObj));
                    }} disabled={search === ""}>Clear</MonthNavSubBtn>
                </div>
            </SearchNavWrapper>
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
const MonthNavSubBtn = styled.button`
    background-color: transparent;
    border: none;
    width: 90%;
    font-size: 16px;
    color: var(--ls-gray_google2);
    cursor: pointer;
    &:hover {
        color: var(--ls-blue);
    }
    &:not(:first-child) {
        border-top: 1px solid var(--ls-gray);
        padding-top: 5px;
    }
    margin-bottom: 3px;
    margin-left: 5px;
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
const SearchNavWrapper = styled.div`
    display: flex;
    align-items: center;

    > div:last-child {
        width: 10%;
    }
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
    grid-template-columns: 4fr 12fr;
    padding-left: 70px;
    padding-right: 70px;

    .TextField {
        width   : 90%;
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
            <span>Tag</span>
            <span>Amount</span>
            <span>Memo</span>
            <div></div>
        </TrxnGridDetailHeaderDiv>
      );
  }else if(viewMode === ViewMode.Graph){
      return (
        <TrxnGridGraphicHeaderDiv>
            <span>Date</span>
            <span>Tag</span>
            <span>Amount</span>
        </TrxnGridGraphicHeaderDiv>
      );
  }else{
    return <></>
  }
};
export function TrxnGridItem({ index, item, isEditing, setEditID, viewMode }: TrxnGridItemProps) {
  const [trxnItem, setTrxnItem] = useState<TrxnElement>(item);

  const dispatch = useDispatch<AppDispatch>();

  if(viewMode === ViewMode.Detail){
    return (<TrxnGridDetailItemDiv key={item.id}>
        <span>{index + 1}</span>
        <span>{GetDateTimeFormatFromDjango(item.date, true)}</span>
        <span>{item.period > 0 ? item.period : '-'}</span>
        <span>{item.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
        
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
            {!isEditing && <RoundButton onClick={async () => { setEditID(item.id); setTrxnItem(item); }}><FontAwesomeIcon icon={faPencil}/></RoundButton>}
            {!isEditing && <RoundButton onClick={async () => {
                if (window.confirm('정말 기록을 삭제하시겠습니까?')) {
                    dispatch(deleteTrxn(item.id));
                }}}><FontAwesomeIcon icon={faX}/></RoundButton>}
        </div>
    </TrxnGridDetailItemDiv>);
  }else if(viewMode === ViewMode.Graph){
    return (<TrxnGridGraphicItemDiv key={item.id}>
        <span>{GetDateTimeFormatFromDjango(item.date, true)}</span>
        <span>{item.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
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
        text-align: center;
        font-size: 22px;
        border-right: 1px solid gray;
        border-bottom: 1px solid gray;
        padding-bottom: 5px;
    }
`;
const TrxnGridDetailItemDiv = styled(TrxnGridDetailTemplate)`
    &:nth-child(4) { 
        /* First Grid Item */
        height: 40px;
        > span {
            padding-top: 5px;
        }
    }
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