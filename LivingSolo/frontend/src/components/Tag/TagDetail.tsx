import styled from "styled-components";
import { TagElement, deleteTag, editTag, selectTag } from "../../store/slices/tag";
import { TagBubble, TagBubbleCompact, TagBubbleHuge } from "../general/TagBubble";
import { useDispatch, useSelector } from "react-redux";
import { GetDateTimeFormatFromDjango } from "../../utils/DateTime";
import { DeleteBtn, EditBtn, EditCompleteBtn } from "../general/FuncButton";
import { useEffect, useState } from "react";
import { AppDispatch } from "../../store";

interface TagDetailProps {
    selectedTag: TagElement | undefined,
    setSelectedTag: React.Dispatch<React.SetStateAction<TagElement | undefined>>
};


// containers/TagMain.tsx에서 사용되는 TagDetail 패널.
export const TagDetail = ({ selectedTag, setSelectedTag } : TagDetailProps) => {
    const dispath = useDispatch<AppDispatch>();
    const { tagDetail } = useSelector(selectTag);
    const [editText, setEditText] = useState<string>('');
    const [editMode, setEditMode] = useState<boolean>(false);

    useEffect(() => {
        selectedTag && setEditText(selectedTag.name);
    }, [selectedTag]);

    const editCompleteHandler = (id: number) => {
        (editText !== selectedTag?.name) && dispath(editTag({ id, name: editText }));
        setSelectedTag(st => { return {...(st as TagElement), name: editText}}); // Fake Updating in SelectedTag Header
        setEditMode(false);
    };
    const deleteHandler = (id: number) => {
        dispath(deleteTag({ id }));
        setSelectedTag(undefined);
    };

    return <>
        {selectedTag && <TagDetailHeaderWrapper>
            <TagBubbleHuge color={selectedTag.color}>{editMode ? <input type="text" value={editText} onChange={(e) => setEditText(e.target.value)} /> : <span>{selectedTag.name}</span>}</TagBubbleHuge>
            <TagDetailHeaderFnWrapper>
                {editMode ? <EditCompleteBtn disabled={editText === ''} handler={() => editCompleteHandler(selectedTag.id)} /> : <EditBtn handler={() => { setEditMode(true); }} />}
                <DeleteBtn confirmText={`정말 ${selectedTag.name} 태그를 삭제하시겠습니까?`} handler={() => { deleteHandler(selectedTag.id) }} />
            </TagDetailHeaderFnWrapper>
        </TagDetailHeaderWrapper>}
        <TodoWrapper>
            <h1>Todo</h1>
            <div>
                {tagDetail && tagDetail.todo.length > 0 && tagDetail.todo.map((todo) => {
                    return <TodoElement key={todo.id}>
                        <span>{GetDateTimeFormatFromDjango(todo.deadline, true)} </span>
                        <TagBubble color={todo.category.color}>{todo.category.name}</TagBubble>
                        {todo.name}
                        <div>
                            {todo.tag.map(t => <TagBubbleCompact color={t.color}>{t.name}</TagBubbleCompact>)}
                        </div>
                    </TodoElement>
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
                    return <TrxnElement key={trxn.id}>
                        <span>{GetDateTimeFormatFromDjango(trxn.date, true)}</span>
                        <span>{trxn.memo}</span>
                        <span>{trxn.amount} 원</span>
                        <div>
                            {trxn.tag.map(t => <TagBubbleCompact color={t.color}>{t.name}</TagBubbleCompact>)}
                        </div>
                    </TrxnElement>
                })}  
            </div>
            {tagDetail && tagDetail.transaction.length === 0 && <>
                <h2>연결된 Transaction이 없어요!</h2>
            </>}
        </TrxnWrapper>
    </>
};

const TagDetailHeaderWrapper = styled.div`
    width: 100%;
    height: fit-content;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
`;
const TagDetailHeaderFnWrapper = styled.div`
    display: flex;
`;

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
        margin: 8px 0px 0px 8px;
    };
    > div {
        padding: 10px 20px;
        width: 100%;

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

const TodoElement = styled.div`
  display: grid;
  grid-template-columns: 3fr 4fr 10fr 6fr;
  align-items: center;

  min-height: 20px;
  width: 100%;
  padding: 5px 0px 0px 0px;

  border-top: 1px solid var(--ls-gray_lighter);
  margin-top: 5px;
  &:first-child {
    border: none;
    margin-top: 0px;
  };
`;
const TrxnElement = styled.div`
  display: grid;
  grid-template-columns: 3fr 4fr 10fr 6fr;
  align-items: center;

  min-height: 20px;
  width: 100%;
  padding: 5px 0px 0px 0px;

  border-top: 1px solid var(--ls-gray_lighter);
  margin-top: 5px;
  &:first-child {
    border: none;
    margin-top: 0px;
  };
`;