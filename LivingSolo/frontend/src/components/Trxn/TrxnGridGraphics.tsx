import { styled } from "styled-components";
import { TrxnElement } from "../../store/slices/trxn";
import { GetDateTimeFormatFromDjango } from "../../utils/DateTime";
import { TagBubbleCompact } from "../general/TagBubble";


interface TrxnGridGraphicItemProps {
    item: TrxnElement,
};

export function TrxnGridGraphicHeader() {
    return (<TrxnGridGraphicHeaderDiv>
        <span>Date</span>
        <span>Tag</span>
        <span>Amount</span>
    </TrxnGridGraphicHeaderDiv>);
};

export function TrxnGridGraphicItem({ item }: TrxnGridGraphicItemProps) {
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
};

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