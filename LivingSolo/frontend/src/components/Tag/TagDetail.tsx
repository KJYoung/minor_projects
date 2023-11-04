import styled from "styled-components";
import { TagElement, deleteTag, editTag, selectTag } from "../../store/slices/tag";
import { TagBubble, TagBubbleCompact, TagBubbleHuge } from "../general/TagBubble";
import { useDispatch, useSelector } from "react-redux";
import { GetDateTimeFormatFromDjango } from "../../utils/DateTime";
import { DeleteBtn, EditBtn, EditCompleteBtn } from "../general/FuncButton";
import { useEffect, useState } from "react";
import { AppDispatch } from "../../store";
import { AutogrowInputWrapper, HiddenTextforAutogrowInput } from "../../utils/Rendering";
import { CharNumSpan } from "../general/FuncSpan";
import { TAG_NAME_LENGTH, TODO_URLS_BY_DJANGO_STRING, TRXN_URLS_BY_DJANGO_STRING } from "../../utils/Constants";
import { notificationWarning } from "../../utils/sendNoti";
import { useNavigate } from "react-router-dom";

interface TagDetailProps {
    selectedTag: TagElement | undefined,
    setSelectedTag: React.Dispatch<React.SetStateAction<TagElement | undefined>>
};


// containers/TagMain.tsx에서 사용되는 TagDetail 패널.
export const TagDetail = ({ selectedTag, setSelectedTag } : TagDetailProps) => {
    const dispath = useDispatch<AppDispatch>();
    const navigate = useNavigate();
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

    const TagNameInputChangeHandler = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newText = e.target.value;
        if(newText.length > TAG_NAME_LENGTH){
            notificationWarning('Tag', `Tag Name Length Should be < ${TAG_NAME_LENGTH}`);
        }else{
            setEditText(newText);
        }
    };
    return <>
        {selectedTag && <TagDetailHeaderWrapper>
            <TagDetailIconWrapperWithCharNum>
                <TagBubbleHuge color={selectedTag.color}>
                    {editMode ? (
                        <AutogrowInputWrapper>
                            <HiddenTextforAutogrowInput>{editText}</HiddenTextforAutogrowInput>
                            <TagHeadInput type="text" value={editText} onChange={TagNameInputChangeHandler} />
                        </AutogrowInputWrapper>) 
                    : 
                        <span>{selectedTag.name}</span>
                    }
                </TagBubbleHuge>
                <CharNumSpan currentNum={editText.length} maxNum={TAG_NAME_LENGTH}/>
            </TagDetailIconWrapperWithCharNum>
            <TagDetailHeaderFnWrapper>
                {editMode ? <EditCompleteBtn disabled={editText === ''} handler={() => editCompleteHandler(selectedTag.id)} /> : <EditBtn handler={() => { setEditMode(true); }} />}
                <DeleteBtn confirmText={`정말 ${selectedTag.name} 태그를 삭제하시겠습니까?`} handler={() => { deleteHandler(selectedTag.id) }} />
            </TagDetailHeaderFnWrapper>
        </TagDetailHeaderWrapper>}
        <TodoWrapper>
            <h1>Todo</h1>
            <div>
                {tagDetail && tagDetail.todo.length > 0 && tagDetail.todo.map((todo) => {
                    return <TodoElement key={todo.id} onClick={() => navigate(TODO_URLS_BY_DJANGO_STRING(todo.deadline))}>
                        <span>{GetDateTimeFormatFromDjango(todo.deadline, true)} </span>
                        <TagBubble color={todo.category.color}>{todo.category.name}</TagBubble>
                        {todo.name}
                        <div>
                            {todo.tag.map(t => <TagBubbleCompact color={t.color} key={t.id}>{t.name}</TagBubbleCompact>)}
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
                    return <TrxnElement key={trxn.id} onClick={() => navigate(TRXN_URLS_BY_DJANGO_STRING(trxn.date))}>
                        <span>{GetDateTimeFormatFromDjango(trxn.date, true)}</span>
                        <span>{trxn.memo}</span>
                        <span>{trxn.amount} 원</span>
                        <div>
                            {trxn.tag.map(t => <TagBubbleCompact color={t.color} key={t.id}>{t.name}</TagBubbleCompact>)}
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

const TagHeadInput = styled.input`
    all: unset;
    border-bottom: 1px solid var(--ls-blue);
    position: absolute;
    width: 100%;
    left: 0;
`;

const TagDetailHeaderWrapper = styled.div`
    width: 100%;
    height: fit-content;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
`;
const TagDetailIconWrapperWithCharNum = styled.div`
    display: flex;
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
  cursor:pointer;
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
  cursor:pointer;
`;