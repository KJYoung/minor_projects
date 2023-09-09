import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { Button } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { SortState, TrxnActions, TrxnElement, TrxnSortState, TrxnSortTarget, deleteTrxn, editTrxn, selectTrxn } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TagBubble';
import { GetDateTimeFormatFromDjango } from '../../utils/DateTime';
import { EditAmountInput } from './AmountInput';
import { ViewMode } from '../../containers/TrxnMain';
import { RoundButton } from '../../utils/Button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPencil, faX } from '@fortawesome/free-solid-svg-icons';
import { TrxnGridGraphicHeader, TrxnGridGraphicItem } from './TrxnGridGraphics';
import { TagInputForGridHeader } from '../Tag/TagInput';
import { TagElement } from '../../store/slices/tag';
import { IPropsActive } from '../../utils/Interfaces';

interface TrxnGridHeaderProps {
    viewMode: ViewMode
};

interface TrxnGridItemProps extends TrxnGridHeaderProps {
    index: number,
    item: TrxnElement,
    isEditing: boolean,
    setEditID: React.Dispatch<React.SetStateAction<number>>
};

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
    return <TrxnGridDetailFilterHeader isActive={(targetState !== SortState.NotSort).toString()} onClick={() => TrxnSortStateHandler(targetEnum)}>
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
            {sortState.tag === SortState.NotSort && <TrxnGridDetailFilterHeader isActive={'false'} onClick={() => TrxnSortStateHandler(TrxnSortTarget.Tag)}>
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
  }else if(viewMode === ViewMode.Combined){
    return (
        <TrxnGridCombinedHeaderDiv className='noselect'>
            {trxnGridDetailFilterHeader(sortState.date, "Date", TrxnSortTarget.Date)}
            {trxnGridDetailFilterHeader(sortState.amount, "Amount", TrxnSortTarget.Amount)}
            
            {/* Unique Logic For Tag Filtering */}
            {sortState.tag === SortState.NotSort && <TrxnGridDetailFilterHeader isActive={'false'} onClick={() => TrxnSortStateHandler(TrxnSortTarget.Tag)}>
                <span>Tag</span>
            </TrxnGridDetailFilterHeader>}
            {sortState.tag === SortState.TagFilter && <div>
                <TagInputForGridHeader tags={tags} setTags={setTags} closeHandler={() => TrxnSortStateHandler(TrxnSortTarget.Tag)}/>
            </div>}

        </TrxnGridCombinedHeaderDiv>
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
const TrxnGridDetailFilterHeader = styled(TrxnGridDetailHeaderItem)<IPropsActive>`
    color: ${props => ((props.isActive === 'true') ? 'var(--ls-blue)' : 'var(--ls-black)')};
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

interface CombinedTrxnGridItemProps {
    index: number,
    date: string,
    amount: number,
    tag: TagElement[],
};

export function CombinedTrxnGridItem({ index, date, amount, tag : tag_ }: CombinedTrxnGridItemProps) {
    if(index === 0 || amount === 0)
        return <></>;

    const tag = tag_.length > 5 ? tag_.slice(0, 5) : tag_;

    return (<TrxnGridCombinedItemDiv key={index}>
        <span>{date}</span>
        <span className='amount'>{amount.toLocaleString()}</span>
        <span>{tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
    </TrxnGridCombinedItemDiv>);
  };

const TrxnGridCombinedTemplate = styled.div`
    display: grid;
    grid-template-columns: 1fr 3fr 2fr 2fr;
    padding-left: 70px;
    padding-right: 70px;   
`;
const TrxnGridCombinedHeaderDiv = styled(TrxnGridCombinedTemplate)`
    min-height: 35px;
`;
const TrxnGridCombinedItemDiv = styled(TrxnGridCombinedTemplate)`
    background-color: var(--ls-gray_lighter2);
    &:nth-child(4) { 
        /* First Grid Item */
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
