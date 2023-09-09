import { useState } from "react";
import { ViewMode } from "../../containers/TrxnMain";
import { CUR_MONTH, CUR_YEAR, CalMonth } from "../../utils/DateTime";
import { TrxnFetchReqType, fetchCombinedTrxns, fetchTrxns } from "../../store/slices/trxn";
import { AppDispatch } from "../../store";
import { useDispatch } from "react-redux";
import { TextField } from "@mui/material";
import { styled } from "styled-components";
import { IPropsActive } from "../../utils/Interfaces";

interface TrxnNavProps {
    viewMode: ViewMode,
    setViewMode: React.Dispatch<React.SetStateAction<ViewMode>>,
    curMonth: CalMonth,
    setCurMonth:  React.Dispatch<React.SetStateAction<CalMonth>>,
}

export function TrxnNav({viewMode, setViewMode, curMonth, setCurMonth} : TrxnNavProps) {
    const [monthMode, setMonthMode] = useState<boolean>(true);
    const [search, setSearch] = useState<string>("");
    const fetchObj: TrxnFetchReqType = {
        yearMonth: CUR_MONTH,
        searchKeyword: "",
    };

    const dispatch = useDispatch<AppDispatch>();
    const TrxnNavCalendar = () => {
        const monthHandler = (am: number) => {
            const afterMonth = (curMonth.month !== undefined ? curMonth.month : 0) + am;

            if(afterMonth > 12)
                fetchObj.yearMonth = {year: curMonth.year + Math.floor(afterMonth / 12), month: afterMonth % 12};
            else if(afterMonth <= 0)
                fetchObj.yearMonth = {year: curMonth.year - Math.floor(afterMonth / 12) - 1, month: 12 - ((-afterMonth) % 12)};
            else
                fetchObj.yearMonth = {...curMonth, month: afterMonth};
            
            dispatch(fetchTrxns(fetchObj));
            dispatch(fetchCombinedTrxns({ yearMonth : fetchObj.yearMonth }));
            setCurMonth(fetchObj.yearMonth);
        };
        const yearHandler = (am: number) => {
            fetchObj.yearMonth = {year: curMonth.year + am};
            setCurMonth(fetchObj.yearMonth);
            dispatch(fetchTrxns(fetchObj));
            dispatch(fetchCombinedTrxns({ yearMonth : fetchObj.yearMonth }));
        };
        const todayHandler = () => {
            fetchObj.yearMonth = monthMode ? CUR_MONTH : CUR_YEAR;
            setCurMonth(fetchObj.yearMonth);
            dispatch(fetchTrxns(fetchObj));
            dispatch(fetchCombinedTrxns({ yearMonth : fetchObj.yearMonth }));
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
    return <TrxnNavWrapper>
        <div>
            <TrxnGridModeBtn isActive={(viewMode === ViewMode.Detail).toString()} onClick={() => setViewMode(ViewMode.Detail)}>Detailed</TrxnGridModeBtn>
            <TrxnGridModeBtn isActive={(viewMode === ViewMode.Combined).toString()} onClick={() => setViewMode(ViewMode.Combined)}>Combined</TrxnGridModeBtn>
            <TrxnGridModeBtn isActive={(viewMode === ViewMode.Graph).toString()} onClick={() => setViewMode(ViewMode.Graph)}>Graphic</TrxnGridModeBtn>
        </div>
        <TrxnGridDetailedSubNav>
            <TrxnNavCalendar />
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
        </TrxnGridDetailedSubNav>
    </TrxnNavWrapper>
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
const TrxnNavWrapper = styled.div`
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
const TrxnGridModeBtn = styled.span<IPropsActive>`
    font-size: 22px;
    margin-left: 20px;
    
    color: ${props => ((props.isActive === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    cursor: ${props => ((props.isActive === 'true') ? 'pointer' : 'default')};
`;
