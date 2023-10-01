import styled from "styled-components";
import { TagElement, selectTag } from "../../store/slices/tag";
import { TagBubbleHuge } from "../general/TagBubble";
import { useSelector } from "react-redux";

interface TagDetailProps {
    selectedTag: TagElement | undefined
};


// containers/TagMain.tsx에서 사용되는 TagDetail 패널.
export const TagDetail = ({ selectedTag } : TagDetailProps) => {
    const { tagDetail } = useSelector(selectTag);

    return <>
        {selectedTag && <TagBubbleHuge color={selectedTag.color}>{selectedTag.name}</TagBubbleHuge>}
        <TodoWrapper>
            <h1>Todo</h1>
            <div>
                
                {tagDetail && tagDetail.todo.length > 0 && tagDetail.todo.map((todo) => {
                    return <div key={todo.id}>{todo.name}</div>
                })}
            </div>
            {tagDetail && tagDetail.todo.length === 0 && <>
                <h2>연결된 Todo가 없어요!</h2>
            </>}
        </TodoWrapper>

        <TrxnWrapper>
            <h1>Transaction</h1>
            <div>
                {tagDetail && tagDetail.transaction.length > 0 && tagDetail?.transaction.map((trxn) => {
                    return <div key={trxn.id}>{trxn.memo}</div>
                })}  
            </div>
            {tagDetail && tagDetail.transaction.length === 0 && <>
                <h2>연결된 Transaction이 없어요!</h2>
            </>}
        </TrxnWrapper>
    </>
};

const AbstractContentWrapper = styled.div`
    width:100%;
    min-height: 60px;
    margin-top: 15px;

    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;

    > h1 {
        font-size: 24px;
        font-weight: 400;
        color: var(--ls-gray_darker1);
        margin: 5px 0px 0px 5px;
    };
    > div {
        padding: 10px 20px;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
    };
    h2 {
        width: 100%;
        font-size: 24px;
        color: var(--ls-gray);
        text-align: center;
        margin: 0px 0px 20px 0px;
    };
    
    background-color: var(--ls-gray_lighter2);
`;
const TodoWrapper = styled(AbstractContentWrapper)`
`;
const TrxnWrapper = styled(AbstractContentWrapper)`
`;