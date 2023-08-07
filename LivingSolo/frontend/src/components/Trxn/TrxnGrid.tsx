import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { Button, TextField } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { SortState, TrxnActions, TrxnElement, TrxnFetchReqType, TrxnSortState, TrxnSortTarget, deleteTrxn, editTrxn, fetchTrxns, selectTrxn } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TagBubble';
import { CUR_MONTH, CUR_YEAR, CalMonth, GetDateTimeFormatFromDjango } from '../../utils/DateTime';
import { EditAmountInput } from './AmountInput';
import { ViewMode } from '../../containers/TrxnMain';
import { RoundButton } from '../../utils/Button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPencil, faX } from '@fortawesome/free-solid-svg-icons';
import { TrxnGridGraphicHeader, TrxnGridGraphicItem } from './TrxnGridGraphics';
import { TagInputForGridHeader } from './TagInput';
import { TagElement } from '../../store/slices/tag';

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

    const searchBtnClickListener = () => {
        fetchObj.searchKeyword = search;
        search !== "" && dispatch(fetchTrxns(fetchObj));
    }
    return <TrxnGridNavWrapper>
        <div>
            <TrxnGridModeBtn active={(viewMode === ViewMode.Detail).toString()} onClick={() => setViewMode(ViewMode.Detail)}>Detailed</TrxnGridModeBtn>
            <TrxnGridModeBtn active={(viewMode === ViewMode.Combined).toString()} onClick={() => setViewMode(ViewMode.Combined)}>Combined</TrxnGridModeBtn>
            <TrxnGridModeBtn active={(viewMode === ViewMode.Graph).toString()} onClick={() => setViewMode(ViewMode.Graph)}>Graphic</TrxnGridModeBtn>
        </div>
        {viewMode === ViewMode.Detail && <TrxnGridDetailedSubNav>
            <TrxnGridNavCalendar />
            <SearchNavWrapper>
                <TextField 
                    className="TextField" label="검색" variant="outlined" value={search} onChange={(e) => setSearch(e.target.value)}
                    onKeyUp={(e) => {
                        if (e.key === 'Enter') {
                            searchBtnClickListener()
                    }}}/>
                <div>
                    <MonthNavSubBtn onClick={() => searchBtnClickListener()}>검색!</MonthNavSubBtn>
                    <MonthNavSubBtn onClick={() => {
                        setSearch("");
                        fetchObj.searchKeyword = "";
                        dispatch(fetchTrxns(fetchObj));
                    }}>Clear</MonthNavSubBtn>
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


const getNextTrxnSortState = (curState: SortState) => {
    switch(curState) {
        case SortState.NotSort:
            return SortState.Descend;
        case SortState.Descend:
            return SortState.Ascend;
        case SortState.Ascend:
            return SortState.NotSort;
        default:
            return SortState.NotSort;
    };
};
const isTrxnSortStateDefault = (curState: TrxnSortState) => {
    return curState.amount === SortState.NotSort && 
        curState.date === SortState.NotSort &&
        curState.memo === SortState.NotSort &&
        curState.period === SortState.NotSort &&
        curState.tag === SortState.NotSort;
};

export function TrxnGridHeader({ viewMode }: TrxnGridHeaderProps ) {
  const dispatch = useDispatch<AppDispatch>();
  const { sortState }  = useSelector(selectTrxn);
  const [tags, setTags] = useState<TagElement[]>([]);

  useEffect(() => {
    dispatch(TrxnActions.setTrxnFilterTag(tags));
  }, [tags, dispatch]);

  const TrxnSortStateHandler = (trxnSortTarget: TrxnSortTarget) => {
    switch(trxnSortTarget) {
        case TrxnSortTarget.Date:
            return dispatch(TrxnActions.setTrxnSort({...sortState, date: getNextTrxnSortState(sortState.date)}));
        case TrxnSortTarget.Period:
            return dispatch(TrxnActions.setTrxnSort({...sortState, period: getNextTrxnSortState(sortState.period)}));
        case TrxnSortTarget.Amount:
            return dispatch(TrxnActions.setTrxnSort({...sortState, amount: getNextTrxnSortState(sortState.amount)}));
        case TrxnSortTarget.Memo:
            return dispatch(TrxnActions.setTrxnSort({...sortState, memo: getNextTrxnSortState(sortState.memo)}));
        case TrxnSortTarget.Tag:
            return dispatch(TrxnActions.setTrxnSort({...sortState, tag: sortState.tag === SortState.NotSort ? SortState.TagFilter : SortState.NotSort}));
    };
  };

  const TrxnSortStateClear = () => {
    dispatch(TrxnActions.clearTrxnSort({}));
    setTags([]);
  };

  const trxnGridDetailFilterHeader = (targetState: SortState, name: string, targetEnum: TrxnSortTarget) => {
    return <TrxnGridDetailFilterHeader active={(targetState !== SortState.NotSort).toString()} onClick={() => TrxnSortStateHandler(targetEnum)}>
        <span>{name}</span>
        <span>
            {targetState === SortState.Descend && '▼'}
            {targetState === SortState.Ascend && '▲'}
        </span>
    </TrxnGridDetailFilterHeader>
  }

  if(viewMode === ViewMode.Detail){
      return (
        <TrxnGridDetailHeaderDiv className='noselect'>
            <TrxnGridDetailHeaderItem>
                <span>Index</span>
            </TrxnGridDetailHeaderItem>
            {trxnGridDetailFilterHeader(sortState.date, "Date", TrxnSortTarget.Date)}
            {trxnGridDetailFilterHeader(sortState.period, "Period", TrxnSortTarget.Period)}

            {/* Unique Logic For Tag Filtering */}
            {sortState.tag === SortState.NotSort && <TrxnGridDetailFilterHeader active={'false'} onClick={() => TrxnSortStateHandler(TrxnSortTarget.Tag)}>
                <span>Tag</span>
            </TrxnGridDetailFilterHeader>}
            {sortState.tag === SortState.TagFilter && <div>
                <TagInputForGridHeader tags={tags} setTags={setTags} closeHandler={() => TrxnSortStateHandler(TrxnSortTarget.Tag)}/>
            </div>}
            
            {trxnGridDetailFilterHeader(sortState.amount, "Amount", TrxnSortTarget.Amount)}
            {trxnGridDetailFilterHeader(sortState.memo, "Memo", TrxnSortTarget.Memo)}
            
            <TrxnGridDetailFilterReset onClick={() => TrxnSortStateClear()}>
                <span>{!isTrxnSortStateDefault(sortState) && 'Reset Filter'}</span>
            </TrxnGridDetailFilterReset>
        </TrxnGridDetailHeaderDiv>
      );
  }else if(viewMode === ViewMode.Graph){
      return <TrxnGridGraphicHeader/>
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
    return (<TrxnGridGraphicItem item={item}/>);
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
    min-height: 35px;
`;
const TrxnGridDetailHeaderItem = styled.div`
    text-align: center;
    font-size: 22px;
    padding-bottom: 5px;

    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0px 20px 0px 20px;
`;
const TrxnGridDetailFilterHeader = styled(TrxnGridDetailHeaderItem)<{ active: string }>`
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-black)')};
    cursor: pointer;
`;
const TrxnGridDetailFilterReset = styled(TrxnGridDetailHeaderItem)`
    font-size: 16px;
    color: var(--ls-gray);
    :hover {
        color: var(--ls-blue);
    }
    cursor: pointer;
`;

const TrxnGridDetailItemDiv = styled(TrxnGridDetailTemplate)`
    background-color: var(--ls-gray_lighter2);
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

