{4.3.0.3419}
{Enterprise Dynamics startup information}

if(StartingED, SoftStartED([]));


{Model information}

AddLayer([Stands], 1, 16, 0);
AddLayer([DoNotDraw], 10, 2, 1);
AddLayer([Output], 1, 15, 2);
AddLayer([Infrastructure], 1, 9, 3);
AddLayer([Agents], 1, 12, 4);
AddLayer([Activities], 1, 11, 5);
AddLayer([Openings], 1, 8, 6);
AddLayer([WalkableAreas], 1, 7, 7);
AddLayer([Obstacles], 1, 6, 8);
AddLayer([ECMNetwork], 27, 4, 9);
AddLayer([HeightLayers], 1, 5, 10);
AddLayer([Main], 1, 1, 11);
AddLayer([PD_Environment], 27, 3, 12);
AddLayer([Resources], 1, 20, 13);
AddLayer([StandsActivity], 1, 17, 14);
AddLayer([Transportation], 1, 18, 15);
AddLayer([LocalObstacles], 1, 14, 16);
AddLayer([UserActions], 1, 13, 17);
AddLayer([Elevators], 3, 10, 18);


{Load required atoms}

StoreActiveAtom;
LoadAtomFromFile([ActivityLocation], pDir([Atoms\ACTIVITIES\ActivityLocation.atm]));
LoadAtomFromFile([AgentDrawInformation], pDir([Atoms\AGENTS\AgentDrawInformation.atm]));
LoadAtomFromFile([AgentProfile], pDir([Atoms\AGENTS\AgentProfile.atm]));
LoadAtomFromFile([HeightLayer], pDir([Atoms\ENVIRONMENT\HeightLayer.atm]));
LoadAtomFromFile([Obstacle], pDir([Atoms\ENVIRONMENT\Obstacle.atm]));
LoadAtomFromFile([PD_Environment], pDir([Atoms\ENVIRONMENT\PD_Environment.atm]));
LoadAtomFromFile([WalkableArea], pDir([Atoms\ENVIRONMENT\WalkableArea.atm]));
LoadAtomFromFile([Experiment_Wizard], pDir([Atoms\EXPERIMENT\Experiment_Wizard.atm]));
LoadAtomFromFile([AgentActivity], pDir([Atoms\INPUT\AgentActivity.atm]));
LoadAtomFromFile([AgentActivityRoute], pDir([Atoms\INPUT\AgentActivityRoute.atm]));
LoadAtomFromFile([AgentGenerator], pDir([Atoms\INPUT\AgentGenerator.atm]));
LoadAtomFromFile([AgentInput], pDir([Atoms\INPUT\AgentInput.atm]));
LoadAtomFromFile([AgentSkeleton], pDir([Atoms\INPUT\AgentSkeleton.atm]));
LoadAtomFromFile([PD_Input], pDir([Atoms\INPUT\PD_Input.atm]));
LoadAtomFromFile([SimControl], pDir([Atoms\INPUT\SimControl.atm]));
LoadAtomFromFile([SkeletonBehaviour], pDir([Atoms\INPUT\SkeletonBehaviour.atm]));
LoadAtomFromFile([SlopeSpeed], pDir([Atoms\INPUT\SlopeSpeed.atm]));
LoadAtomFromFile([ActivityRoute], pDir([Atoms\OUTPUT\ActivityRoute.atm]));
LoadAtomFromFile([AgentStatistics], pDir([Atoms\OUTPUT\AgentStatistics.atm]));
LoadAtomFromFile([DensityNorms], pDir([Atoms\OUTPUT\DensityNorms.atm]));
LoadAtomFromFile([FlowCounter], pDir([Atoms\OUTPUT\FlowCounter.atm]));
LoadAtomFromFile([FrequencyNorms], pDir([Atoms\OUTPUT\FrequencyNorms.atm]));
LoadAtomFromFile([lstOutputActivityRoutes], pDir([Atoms\OUTPUT\lstOutputActivityRoutes.atm]));
LoadAtomFromFile([OutputLayer], pDir([Atoms\OUTPUT\OutputLayer.atm]));
LoadAtomFromFile([PD_Output], pDir([Atoms\OUTPUT\PD_Output.atm]));
LoadAtomFromFile([ProximityNorms], pDir([Atoms\OUTPUT\ProximityNorms.atm]));
LoadAtomFromFile([ResultPlayer], pDir([Atoms\OUTPUT\ResultPlayer.atm]));
LoadAtomFromFile([TravelTimeNorms], pDir([Atoms\OUTPUT\TravelTimeNorms.atm]));
LoadAtomFromFile([CameraPositions], pDir([Atoms\TOOLS\CameraPositions.atm]));
LoadAtomFromFile([DisplayAtom], pDir([Atoms\TOOLS\DisplayAtom.atm]));
LoadAtomFromFile([MovieRecord], pDir([Atoms\TOOLS\MovieRecord.atm]));
LoadAtomFromFile([RunClock], pDir([Atoms\TOOLS\RunClock.atm]));
LoadAtomFromFile([SelectionPolygon], pDir([Atoms\TOOLS\SelectionPolygon.atm]));
LoadAtomFromFile([StaticAnalysis], pDir([Atoms\TOOLS\StaticAnalysis.atm]));
LoadAtomFromFile([UserEvents], pDir([Atoms\TOOLS\UserEvents.atm]));
LoadAtomFromFile([List], pDir([Atoms\TOOLS\UTILITIES\List.atm]));
LoadAtomFromFile([TransportGenerator], pDir([Atoms\TRANSPORTATION\TransportGenerator.atm]));
LoadAtomFromFile([TransportInput], pDir([Atoms\TRANSPORTATION\TransportInput.atm]));
LoadAtomFromFile([TransportOutput], pDir([Atoms\TRANSPORTATION\TransportOutput.atm]));
LoadAtomFromFile([TransportSubType], pDir([Atoms\TRANSPORTATION\TransportSubType.atm]));
LoadAtomFromFile([TransportType], pDir([Atoms\TRANSPORTATION\TransportType.atm]));
RestoreActiveAtom;


{Atom: PD_Input}

sets;
AtomByName([PD_Input], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Input'. Inheriting from BaseClass.]));
CreateAtom(a, s, [PD_Input], 1, false);
SetAtt([lstTempAgents], 0);
SetExprAtt([InitCode], [0]);
SetAtt([lstTables], 0);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 530432);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(1);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstTables}

sets;
AtomByName([lstTables], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTables'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstTables], 1, false);
SetAtt([NrCreated], 0);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(2);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: SimControl}

sets;
AtomByName([SimControl], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SimControl'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SimControl], 1, false);
SetAtt([CycleTime], 0.2);
SetAtt([UseDensityPlanning], 1);
SetAtt([AvgAgentArea], 0.18);
SetAtt([RebuildECMNetwork], 1);
SetAtt([UNUSED_DensityDistance], 2);
SetAtt([ECMLoaded], 0);
SetAtt([PixelMeter], 0.05);
SetAtt([AgentColorType2D], 10002);
SetAtt([AgentColorType3D], 10002);
SetAtt([AgentDrawType2D], 11002);
SetAtt([AgentDrawType3D], 12002);
SetAtt([CycleEventScheduled], 0);
SetExprAtt([StopTimeSeconds], [1]);
SetAtt([RunUntilStopTime], 0);
SetAtt([Initialized], 0);
SetAtt([EvacuationMode], 0);
SetAtt([RouteFollowingMethod], 15001);
SetAtt([AvoidCollisions_DensityThreshold], -1);
SetAtt([VisualizeSpeed], 1);
SetAtt([UseMesoDensitySpeed], 1);
SetAtt([StartingDrawType], 12002);
SetAtt([AgentPrivateCircleType], 2);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(3);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(3, 5);
CreateTableColumn(0, 64, [`ID`
1
2
3
]);
CreateTableColumn(1, 64, [`Name`
`Normal`
`StairsUp`
`StairsDown`
]);
CreateTableColumn(2, 98, [`FormulaType`
2
2
2
]);
CreateTableColumn(3, 102, [`MaxSpeed`
1.34
0.61
0.694
]);
CreateTableColumn(4, 64, [`dFactor`
1.913
3.722
3.802
]);
CreateTableColumn(5, 64, [`dJam`
5.4
5.4
5.4
]);
SetStatus(0);
ExecLateInit;


{Atom: TransportInput}

sets;
AtomByName([TransportInput], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportInput'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [TransportInput], 1, false);
SetAtt([lstGenerators], 0);
SetAtt([lstTransportSubTypes], 0);
SetAtt([lstTransportTypes], 0);
SetAtt([NetworkID], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(4);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstTransportGenerators}

sets;
AtomByName([lstTransportGenerators], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportGenerators'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstTransportGenerators], 1, false);
SetAtt([NrCreated], 1);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(5);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: TransportGenerator_1}

sets;
AtomByName([TransportGenerator], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportGenerator'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TransportGenerator_1], 1, false);
SetAtt([ID], 1);
SetAtt([RepetitiveMode], 2001);
SetExprAtt([RepeatTime], [0]);
SetExprAtt([OffSetTime], [0]);
SetExprAtt([NrTimesToRepeat], [{**Unlimited**} 0]);
SetExprAtt([DelayTime], [0]);
SetExprAtt([CreationTrigger], [0]);
SetExprAtt([MaxNumberOfTransportElements], [{**Unlimited**} -1]);
SetExprAtt([FindNextCondition], [{**Use LineID and Direction. s or i = agent**}
And(
  CompareText(TransportGenerator_GetLineID(c, ROWINDEX), TransportGenerator_GetLineID(c, Agent_GetTransportGeneratorRow(s))),
  TransportGenerator_GetDirection(c, ROWINDEX) = TransportGenerator_GetDirection(c, Agent_GetTransportGeneratorRow(s)),
  Or(
    Agent_GetLastVisitedDestination(s) = 0, {**Arriving agent, at platform already checked**}
    TransportGenerator_GetDepartureTime(c, ROWINDEX) >= Time + 10 {**At least 10 seconds before departure**}
  )
)]);
SetExprAtt([DepartureCondition], [{**Leave at departure time without waiting for approaching agents**}0]);
SetAtt([ReadDB], 0);
SetExprAtt([DBConnectString], [4DS[Concat([Provider=Microsoft.ACE.OLEDB.12.0;Data Source=], ModDir([PD_Transport_Input.accdb]), [;Persist Security Info=False])]4DS]);
SetTextAtt([TableName], [PD_TrainTimetable]);
SetTextAtt([TableNameTransportTypes], [PD_TransportTypes]);
SetTextAtt([TableNameTransportSubTypes], [PD_TransportSubTypes]);
SetExprAtt([ArrivalTrigger], [0]);
SetAtt([UseCapacity], 1);
SetAtt([ApproachingTime], 0);
SetExprAtt([DepartureTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetMultiVectorCompareByFields(0, 2);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(6);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 16);
CreateTableColumn(0, 64, [`Row`
1
]);
CreateTableColumn(1, 137, [`ID`
`Train1`
]);
CreateTableColumn(2, 63, [`LineID`
`Line1`
]);
CreateTableColumn(3, 61, [`ArrivalTime`
3600
]);
CreateTableColumn(4, 76, [`DepartureTime`
3660
]);
CreateTableColumn(5, 99, [`TransportType`
1
]);
CreateTableColumn(6, 124, [`Caption`
`MyTrain`
]);
CreateTableColumn(7, 120, [`NetworkID`
1
]);
CreateTableColumn(8, 120, [`StopLocation`
90
]);
CreateTableColumn(9, 52, [`EntrySide`
1
]);
CreateTableColumn(10, 46, [`ExitSide`
1
]);
CreateTableColumn(11, 51, [`Direction`
1
]);
CreateTableColumn(12, 108, [`Boarding Distribution`
`dUniform(1, TransportElement_GetNrSubTypes(c))`
]);
CreateTableColumn(13, 64, [`DeBoarding Distribution`
`dUniform(1, TransportElement_GetNrSubTypes(c))`
]);
CreateTableColumn(14, 64, [`DelayTime`
0
]);
CreateTableColumn(15, 64, [`Initial content`
0
]);
CreateTableColumn(16, 64, [`Minimum dwell time`
0
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstTransportSubTypes}

sets;
AtomByName([lstTransportSubTypes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportSubTypes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTransportSubTypes], 1, false);
SetAtt([NrCreated], 1);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(7);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: TransportSubType_A}

sets;
AtomByName([TransportSubType], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportSubType'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TransportSubType_A], 1, false);
SetAtt([NrUnscheduledAgents], 0);
SetAtt([NrDeBoarding], 0);
SetAtt([NrFoundPlatforms], 0);
SetExprAtt([OutlineColor], [ColorBlue]);
SetExprAtt([DoorColor], [ColorOrange]);
SetAtt([InnerDistance], 0.5);
SetAtt([ID], 1);
SetAtt([Capacity], 50);
SetAtt([Draw3DModel], 0);
SetExprAtt([Model3DID], [0]);
SetDataContainerItem(0, 1, [0;0;0;1;1;1;0;0;0;
]);
SetMultiVectorCompareByFields(0, 0, 1);
SetAtomSettings([], 3394815, 12336);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(8);
SetSize(10, 2, 2);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 11);
CreateTableColumn(0, 64, [0
1
2
3
4
]);
CreateTableColumn(1, 121, [`Door Position (%)`
25
75
25
75
]);
CreateTableColumn(2, 72, [`Door_Side`
1
1
2
2
]);
CreateTableColumn(3, 64, [`Door_Width`
1
1
1
1
]);
CreateTableColumn(4, 64, [`CapacityBoarding`
50
50
50
50
]);
CreateTableColumn(5, 64, [`CapacityDeBoarding`
50
50
50
50
]);
CreateTableColumn(6, 64, [`WA`
]);
CreateTableColumn(7, 64, [`WALineNr`
]);
CreateTableColumn(8, 64, [`XPOSWA`
]);
CreateTableColumn(9, 64, [`YPOSWA`
]);
CreateTableColumn(10, 64, [`MaxDepth`
]);
CreateTableColumn(11, 64, [`Boarding`
0
0
0
0
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstTransportTypes}

sets;
AtomByName([lstTransportTypes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportTypes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTransportTypes], 1, false);
SetAtt([NrCreated], 1);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(9);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: TransportType_1}

sets;
AtomByName([TransportType], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportType'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TransportType_1], 1, false);
SetAtt([ReadDB], 0);
SetExprAtt([DBConnectString], [4DS[Concat([Provider=Microsoft.ACE.OLEDB.12.0;Data Source=], ModDir([PD_Transport_Input.accdb]), [;Persist Security Info=False])]4DS]);
SetTextAtt([TableName], [PD_TransportTypes]);
SetAtt([MaxSpeed], 10);
SetAtt([ID], 1);
SetAtomSettings([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(10);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 2);
CreateTableColumn(0, 64, [0
1
]);
CreateTableColumn(1, 161, [`SubType`
1
]);
CreateTableColumn(2, 149, [`Number`
2
]);
SetStatus(0);
ExecLateInit;
Up;
Up;


{Atom: AgentInput}

sets;
AtomByName([AgentInput], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentInput'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentInput], 1, false);
SetAtt([lstGenerators], 0);
SetAtt([lstProfiles], 0);
SetAtt([lstActivityRoutes], 0);
SetAtt([lstActivities], 0);
SetAtt([MultiplierMaxValue], 5);
SetAtt([MultiplierStepSize], 0.1);
SetAtt([lstProxies], 0);
SetExprAtt([LastProfileFolder], [4DS[pDir([Settings\])]4DS]);
SetAtt([lstAgentSkeleton], 0);
SetAtt([lstSkeletonBehaviour], 0);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(11);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstGenerators}

sets;
AtomByName([lstGenerators], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstGenerators'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstGenerators], 1, false);
SetAtt([NrCreated], 2);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(12);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: entry1_Generator}

sets;
AtomByName([AgentGenerator], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentGenerator'. Inheriting from BaseClass.]));
CreateAtom(a, s, [entry1_Generator], 1, false);
SetAtt([ID], 1);
SetAtt([RepetitiveMode], 2002);
SetAtt([RepeatTime], 0);
SetExprAtt([OffSetTime], [{**Negative exponentially distributed with mean e1**} NegExp(0.15)]);
SetExprAtt([NrTimesToRepeat], [{**Unlimited**} 0]);
SetAtt([DelayTime], 0);
SetAtt([CreationTrigger], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([MaxNumberOfAgents], [{**Unlimited**} -1]);
SetAtt([CurrentMultiplier], 1);
SetExprAtt([TransportGeneratorID], [0]);
SetAtt([TransportGenerator], 0);
SetAtt([ReadDB], 0);
SetAtt([DBConnectString], 0);
SetAtt([TableName], 0);
SetExprAtt([ResetCode], [0]);
SetAtt([tempMaxNumberOfAgents], -1);
SetAtt([tempCurrentMultiplier], 1);
SetAtt([AgentsAsPercOfMax], 0);
SetAtt([DestroyTrigger], 0);
SetAtomSettings([], 7168771, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
Set(DdbRec, [>basetime1:0.362094598035031.]);
SetChannels(1, 1);
SetChannelRanges(1, 1, 1, 255);
SetLoadAtomID(13);
SetSize(5, 2, 2);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 7);
CreateTableColumn(0, 64, [`Row`
1
]);
CreateTableColumn(1, 64, [`Creation time`
0
]);
CreateTableColumn(2, 64, [`Nr Agents`
1
]);
CreateTableColumn(3, 64, [`Activity route`
`2 {**entry1exit**}`
]);
CreateTableColumn(4, 64, [`Agent profile`
1
]);
CreateTableColumn(5, 64, [`Creation trigger`
0
]);
CreateTableColumn(6, 0, [6
0
]);
CreateTableColumn(7, 0, [7
0
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstProfiles}

sets;
AtomByName([lstProfiles], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstProfiles'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstProfiles], 1, false);
SetAtt([NrCreated], 2);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(14);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Default_Profile}

sets;
AtomByName([AgentProfile], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentProfile'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Profile], 1, false);
SetAtt([ID], 1);
SetExprAtt([MaxSpeed], [Triangular(1.35, 0.8, 1.75)]);
SetExprAtt([MinSpeed], [0.06]);
SetExprAtt([Radius], [0.239]);
SetExprAtt([DensityCostWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([DensityPlanDistance], [60]);
SetExprAtt([ReplanFrequency], [40]);
SetExprAtt([HistoryPenalty], [0]);
SetExprAtt([HeuristicMultiplier], [Uniform(1, 1.15)]);
SetExprAtt([SidePreference], [Bernoulli(50, Uniform(-1, 0), Uniform(0, 1)) {**50% chance leftside spread from -1 to 0, 80% rightside from 0 to 1**}]);
SetExprAtt([SidePreferenceNoise], [0.1]);
SetExprAtt([PreferredClearance], [0.3]);
SetExprAtt([Color], [16711680]);
SetExprAtt([ModelID3D], [0]);
SetExprAtt([DiscomfortWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([FixedSpeedFactor], [0]);
SetAtt([UseShortestPath], 0);
SetExprAtt([PersonalDistance], [0.5]);
SetExprAtt([FieldOfViewAngle], [75]);
SetExprAtt([FieldOfViewCollisionRange], [8]);
SetExprAtt([FieldOfViewDensityRange], [2]);
SetAtt([UseDensityRouting], 1);
SetAtt([UseLimitedDensityPlanDistance], 1);
SetAtt([UseRerouting], 1);
SetExprAtt([RandomExtraEdgeCost], [5]);
SetExprAtt([Tokens], [0]);
SetExprAtt([AvoidanceSidePreference], [{**Right preference**}
0.1]);
SetExprAtt([SideClearanceFactor], [{**Random value between 0 and 1**}
Uniform(0, 1)]);
SetExprAtt([MaxShortCutDistance], [0]);
SetExprAtt([SidePreferenceUpdateFactor], [1]);
SetAtt([RoutingMethod], 0);
SetExprAtt([AggressivenessThreshold], [0.25]);
SetTextAtt([Description], []);
SetAtt([SocialDistancing], 0);
SetExprAtt([PhysicalDistance], [{**No physical/social distancing**} 0]);
SetExprAtt([RelaxationTime], [0.5]);
SetExprAtt([ColorPhysicalDistance], [255]);
SetAtt([SkeletonGenderId], 0);
SetAtt([SkeletonAgeId], 0);
SetAtt([SkeletonStateMachineId], 0);
SetAtt([AgentSkeleton], 0);
SetMultiVectorCompareByFields(0, 0, 1);
SetAtomSettings([], 32768, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(15);
SetSize(0.4, 0.4, 2);
SetTranslation(-0.2, -0.2, 0);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Default_Profile_Copy}

sets;
AtomByName([AgentProfile], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentProfile'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Default_Profile_Copy], 1, false);
SetAtt([ID], 2);
SetExprAtt([MaxSpeed], [Triangular(1.35, 0.8, 1.75)]);
SetExprAtt([MinSpeed], [0.06]);
SetExprAtt([Radius], [0.239]);
SetExprAtt([DensityCostWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([DensityPlanDistance], [60]);
SetExprAtt([ReplanFrequency], [40]);
SetExprAtt([HistoryPenalty], [0]);
SetExprAtt([HeuristicMultiplier], [Uniform(1, 1.15)]);
SetExprAtt([SidePreference], [Bernoulli(50, Uniform(-1, 0), Uniform(0, 1)) {**50% chance leftside spread from -1 to 0, 80% rightside from 0 to 1**}]);
SetExprAtt([SidePreferenceNoise], [0.1]);
SetExprAtt([PreferredClearance], [0.3]);
SetExprAtt([Color], [255]);
SetExprAtt([ModelID3D], [0]);
SetExprAtt([DiscomfortWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([FixedSpeedFactor], [0]);
SetAtt([UseShortestPath], 0);
SetExprAtt([PersonalDistance], [0.5]);
SetExprAtt([FieldOfViewAngle], [75]);
SetExprAtt([FieldOfViewCollisionRange], [8]);
SetExprAtt([FieldOfViewDensityRange], [2]);
SetAtt([UseDensityRouting], 1);
SetAtt([UseLimitedDensityPlanDistance], 1);
SetAtt([UseRerouting], 1);
SetExprAtt([RandomExtraEdgeCost], [5]);
SetExprAtt([Tokens], [0]);
SetExprAtt([AvoidanceSidePreference], [{**Right preference**}
0.1]);
SetExprAtt([SideClearanceFactor], [{**Random value between 0 and 1**}
Uniform(0, 1)]);
SetExprAtt([MaxShortCutDistance], [0]);
SetExprAtt([SidePreferenceUpdateFactor], [1]);
SetAtt([RoutingMethod], 0);
SetExprAtt([AggressivenessThreshold], [0.25]);
SetTextAtt([Description], []);
SetAtt([SocialDistancing], 0);
SetExprAtt([PhysicalDistance], [{**Uniformly distributed between e1 and e2**} Uniform(5, 15)]);
SetExprAtt([RelaxationTime], [0.5]);
SetExprAtt([ColorPhysicalDistance], [255]);
SetAtt([SkeletonGenderId], 0);
SetAtt([SkeletonAgeId], 0);
SetAtt([SkeletonStateMachineId], 0);
SetAtt([AgentSkeleton], 0);
SetMultiVectorCompareByFields(0, 0, 1);
SetAtomSettings([], 13158600, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(16);
SetSize(0.4, 0.4, 2);
SetTranslation(-0.2, -0.2, 0);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstProxies}

sets;
AtomByName([lstProxies], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstProxies'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstProxies], 1, false);
SetAtt([NrCreated], 0);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(17);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstActivityRoutes}

sets;
AtomByName([lstActivityRoutes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstActivityRoutes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstActivityRoutes], 1, false);
SetAtt([NrCreated], 3);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(18);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Default_Route}

sets;
AtomByName([AgentActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Route], 1, false);
SetAtt([ID], 1);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(19);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 2);
CreateTableColumn(0, 0, [``
1
2
]);
CreateTableColumn(1, 0, [1
1
2
]);
CreateTableColumn(2, 0, [2
638843056
871553472
]);
SetStatus(0);
ExecLateInit;


{Atom: entry1exit}

sets;
AtomByName([AgentActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [entry1exit], 1, false);
SetAtt([ID], 2);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(20);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 2);
CreateTableColumn(0, 0, [``
1
2
]);
CreateTableColumn(1, 0, [1
3
4
]);
CreateTableColumn(2, 0, [2
871537632
870548112
]);
SetStatus(0);
ExecLateInit;


{Atom: entry3exit}

sets;
AtomByName([AgentActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [entry3exit], 1, false);
SetAtt([ID], 3);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(21);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 2);
CreateTableColumn(0, 0, [``
1
2
]);
CreateTableColumn(1, 0, [1
5
6
]);
CreateTableColumn(2, 0, [2
871545024
871546080
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstAgentActivities}

sets;
AtomByName([lstAgentActivities], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstAgentActivities'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstAgentActivities], 1, false);
SetAtt([NrCreated], 6);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(22);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Entry}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Entry], 1, false);
SetAtt([ID], 1);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 1);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(23);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
25
25
25
25
]);
SetStatus(0);
ExecLateInit;


{Atom: Exit}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Exit], 1, false);
SetAtt([ID], 2);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(24);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
25
25
25
25
]);
SetStatus(0);
ExecLateInit;


{Atom: Entry1}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Entry1], 1, false);
SetAtt([ID], 3);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(25);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
100
0
0
0
]);
SetStatus(0);
ExecLateInit;


{Atom: Exit3}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Exit3], 1, false);
SetAtt([ID], 4);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(26);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
0
0
100
0
]);
SetStatus(0);
ExecLateInit;


{Atom: Entry3}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Entry3], 1, false);
SetAtt([ID], 5);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(27);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
0
0
100
0
]);
SetStatus(0);
ExecLateInit;


{Atom: Exit1}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Exit1], 1, false);
SetAtt([ID], 6);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_EMPIRICAL]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(28);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(4, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
]);
CreateTableColumn(1, 0, [`ActivityID`
1
2
3
4
]);
CreateTableColumn(2, 0, [`vtp`
873513696
873512640
873511584
873507360
]);
CreateTableColumn(3, 0, [`%`
100
0
0
0
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstSkeletons}

sets;
AtomByName([lstSkeletons], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstSkeletons'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstSkeletons], 1, false);
SetAtt([NrCreated], 4);
SetDataContainerItem(0, 5, [0;1;1;0;0;0;0;0;
0;1;1;1;1;0;0;0;
0;0;0;0;1;0;0;0;
1;0;1;1;0;0;0;0;
1;0;0;0;0;0;0;0;
]);
SetMultiVectorCompareByFields(0, 0, 1);
SetAtomSettings([], 0, 530432);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(29);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Amelia}

sets;
AtomByName([AgentSkeleton], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentSkeleton'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Amelia], 1, false);
SetAtt([ID], 1);
SetTextAtt([Name], [Amelia]);
SetAtt([Gender], 3);
SetAtt([StateMachine], 0);
SetAtt([Age], 3);
SetTextAtt([ModelName], [13C8E78EFFEA2241DDD7CE0CEE5ABAB1]);
SetTextAtt([ModelFilePath], [Media\3DModels\Skeletons\Amelia\Amelia.fbx]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(30);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Lucas}

sets;
AtomByName([AgentSkeleton], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentSkeleton'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Lucas], 1, false);
SetAtt([ID], 2);
SetTextAtt([Name], [Lucas]);
SetAtt([Gender], 2);
SetAtt([StateMachine], 0);
SetAtt([Age], 3);
SetTextAtt([ModelName], [F72FCA5A7EF858188F5B82A61A711701]);
SetTextAtt([ModelFilePath], [Media\3DModels\Skeletons\Lucas\Lucas.fbx]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(31);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Audre}

sets;
AtomByName([AgentSkeleton], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentSkeleton'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Audre], 1, false);
SetAtt([ID], 3);
SetTextAtt([Name], [Audre]);
SetAtt([Gender], 3);
SetAtt([StateMachine], 0);
SetAtt([Age], 3);
SetTextAtt([ModelName], [14DF032E71FE02F18D1AEF22D5BAFB8C]);
SetTextAtt([ModelFilePath], [Media\3DModels\Skeletons\Audre\Audre.fbx]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(32);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Ethan}

sets;
AtomByName([AgentSkeleton], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentSkeleton'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Ethan], 1, false);
SetAtt([ID], 4);
SetTextAtt([Name], [Ethan]);
SetAtt([Gender], 2);
SetAtt([StateMachine], 0);
SetAtt([Age], 3);
SetTextAtt([ModelName], [A8B73C7AB01F433E4C1CA07B37F2A3D6]);
SetTextAtt([ModelFilePath], [Media\3DModels\Skeletons\Ethan\Ethan.fbx]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(33);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstSkeletonBehaviours}

sets;
AtomByName([lstSkeletonBehaviours], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstSkeletonBehaviours'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstSkeletonBehaviours], 1, false);
SetAtt([NrCreated], 3);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(34);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(18, 3);
CreateTableColumn(0, 64, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
]);
CreateTableColumn(1, 64, [1
`scblocationassignment`
`scbnragents`
`scbdelaytime`
`scbcreationtrigger`
`scbexittrigger`
`scbresetcode`
`scbexittrigger`
`scbnragents`
`scblocationassignment`
`scbcreationtime`
`scbprecondition`
`scbpostcondition`
`scblocationfrom`
`scboffsettime`
`scbentrytrigger`
`scbexittrigger`
`scbentrytrigger`
`scbexittrigger`
]);
CreateTableColumn(2, 64, [2
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Stadium_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
`Transportation_GetDistributionFunctions(t)`
]);
CreateTableColumn(3, 64, [3
604968160
604968160
604968160
604968160
604968160
604968160
605181264
604968160
604968160
604968160
604968160
604968160
605181264
604968160
604968160
604968160
605181264
605181264
]);
SetStatus(0);
ExecLateInit;


{Atom: Default_Behaviour1}

sets;
AtomByName([SkeletonBehaviour], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SkeletonBehaviour'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Behaviour1], 1, false);
SetAtt([ID], 1);
SetTextAtt([Name], [Default_Behaviour1]);
SetTextAtt([BehaviourFilePath], [Media\StateMachines\Machine_One.xml]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(35);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Default_Behaviour2}

sets;
AtomByName([SkeletonBehaviour], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SkeletonBehaviour'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Default_Behaviour2], 1, false);
SetAtt([ID], 2);
SetTextAtt([Name], [Default_Behaviour2]);
SetTextAtt([BehaviourFilePath], [Media\StateMachines\Machine_Two.xml]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(36);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Default_Behaviour3}

sets;
AtomByName([SkeletonBehaviour], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SkeletonBehaviour'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Default_Behaviour3], 1, false);
SetAtt([ID], 3);
SetTextAtt([Name], [Default_Behaviour3]);
SetTextAtt([BehaviourFilePath], [Media\StateMachines\Machine_Three.xml]);
SetAtomSettings([], 0, 1162);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(37);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;
Up;


{Atom: SlopeSpeed}

sets;
AtomByName([SlopeSpeed], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SlopeSpeed'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SlopeSpeed], 1, false);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(38);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 3);
CreateTableColumn(0, 64, [`ID`
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`Slope (%)`
0
5
10
15
20
40
]);
CreateTableColumn(2, 98, [`% Speed up`
100
96.3
88.8
79.9
70.9
41
]);
CreateTableColumn(3, 102, [`% Speed down`
100
103
104.5
104.5
103
74.6
]);
SetStatus(0);
ExecLateInit;


{Atom: Experiment_Wizard}

sets;
AtomByName([Experiment_Wizard], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Experiment_Wizard'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Experiment_Wizard], 1, false);
SetAtt([NumberScenarios], 1);
SetAtt([CurScen], 2);
SetAtt([NumberOfReps], 1);
SetExprAtt([Runlength], [hr(2)]);
SetExprAtt([CodeAtStart], [0]);
SetExprAtt([CodeAtEnd], [0]);
SetTextAtt([Path], [C:\Users\13271\Downloads\PD\Data\sc_crossroad\]);
SetAtt([numparams], 1);
SetAtt([executescenario], 1);
SetAtt([LogOutput], 1);
SetAtt([LoadOutputOfLastReplication], 1);
SetAtt([FootstepFile], 0);
SetAtt([LogFootSteps], 1);
SetAtt([LogFootStepFrequency], 1);
SetAtt([Initialized], 1);
SetExprAtt([AddTextToFolderName], [4DS[[]]4DS]);
SetAtt([RunRepetitive], 0);
SetAtomSettings([], 0, 525544);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
Set(DdbRec, [>paramrow:1.>mode:2.>scenariorow:1.]);
SetChannels(1, 0);
SetChannelRanges(0, 1, 0, 0);
SetLoadAtomID(39);
SetChannelRequestCount(1, 0, true, true, 5, 0, [Optional connection to Central Channel of 70 MB Experiment Atom], []);
SetSize(11, 2, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 10);
CreateTableColumn(0, 64, [`Seed_Value`
5221
]);
CreateTableColumn(1, 95, [`Scen nr`
`Crossroads1`
]);
CreateTableColumn(2, 93, [`pad`
`D:\ÏÂÔØ\Devs\Ped\sc_crossroad\`
]);
CreateTableColumn(3, 75, [`Omschrijving`
`--- describe new scenario ---`
]);
CreateTableColumn(4, 90, [`NumReps`
1
]);
CreateTableColumn(5, 79, [`Runlengte`
`mins(30)`
]);
CreateTableColumn(6, 64, [`StartCode`
0
]);
CreateTableColumn(7, 64, [`EndCode`
0
]);
CreateTableColumn(8, 64, [`Execute`
1
]);
CreateTableColumn(9, 64, [`StartRep`
1
]);
CreateTableColumn(10, 0, [`Avoid Collisions`
1
]);
SetStatus(0);
ExecLateInit;


{Atom: EW_ModelParameters}

sets;
AtomByName([EW_ModelParameters], Main);
if(not(AtomExists), Error([Cannot find mother atom 'EW_ModelParameters'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EW_ModelParameters], 1, false);
SetAtomSettings([], 0, 3080);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(40);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 5);
CreateTableColumn(0, 0, [``
`Avoid Collisions`
]);
CreateTableColumn(1, 0, [`defaultwaarde`
0
]);
CreateTableColumn(2, 0, [`type`
2
]);
CreateTableColumn(3, 0, [`Code`
`SimControl.RouteFollowingMethod(atmSimControl) := if(valModelPar, ROUTEFOLLOWINGMETHOD_MICRO, ROUTEFOLLOWINGMETHOD_MESO)`
]);
CreateTableColumn(4, 0, [`VarName`
`valModelPar`
]);
CreateTableColumn(5, 0, [`Run After Reset`
0
]);
SetStatus(0);
ExecLateInit;


{Atom: EW_Experiment}

sets;
AtomByName([EW_Experiment], Main);
if(not(AtomExists), Error([Cannot find mother atom 'EW_Experiment'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EW_Experiment], 1, false);
SetExprAtt([RunLength], [mins(30)]);
SetAtt([nruns], 1);
SetAtt([curRun], 1);
SetTextAtt([path], [D:\ÏÂÔØ\Devs\Ped\sc_crossroad\Crossroads1\]);
SetAtt([isRunningAsExp], 0);
SetExprAtt([OnResetCode], [0]);
SetExprAtt([AfterRunCode], [0]);
SetAtt([StartRep], 0);
SetTextAtt([CurPath], [0]);
SetAtt([StartTime], 3911916234.2);
SetAtt([MultiLogTimes], 0);
SetAtt([MultiLogs], 0);
SetAtomSettings([], 15, 527592);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetChannelRanges(0, 0, 0, 1);
SetLoadAtomID(41);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: UserEvents}

sets;
AtomByName([UserEvents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'UserEvents'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [UserEvents], 1, false);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffSetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetAtomSettings([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(42);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstTempAgents}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTempAgents], 1, false);
SetAtomSettings([], 0, 1024);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(43);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: CameraPositions}

sets;
AtomByName([CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom 'CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [CameraPositions], 1, false);
SetAtt([Default2D_Index], 1);
SetAtt([Default3D_Index], 1);
SetAtt([AtomToFollow], 0);
SetAtt([obj2D_CameraPositions], 0);
SetAtt([obj3D_CameraPositions], 0);
SetExprAtt([LoadComboBoxFunction], [CameraPositions_AddCameraPositionsToComboBox(MAINMENUFORM)]);
SetAtomSettings([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(44);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
CreateTableColumn(0, 64, [`ID`
1
]);
CreateTableColumn(1, 64, [`Show`
]);
CreateTableColumn(2, 0, [`Name`
]);
CreateTableColumn(3, 0, [`2D3D`
]);
CreateTableColumn(4, 0, [`Visible Height layers`
]);
CreateTableColumn(5, 0, [`Window Width`
]);
CreateTableColumn(6, 0, [`Window Heigth`
]);
CreateTableColumn(7, 0, [`Window Xloc`
]);
CreateTableColumn(8, 0, [`Window Yloc`
]);
CreateTableColumn(9, 0, [`Scale`
]);
CreateTableColumn(10, 0, [`ViewX`
]);
CreateTableColumn(11, 0, [`ViewY`
]);
CreateTableColumn(12, 0, [`ViewZ`
]);
CreateTableColumn(13, 0, [`Roll`
]);
CreateTableColumn(14, 0, [`Pitch`
]);
CreateTableColumn(15, 0, [`Yaw`
]);
SetStatus(0);
ExecLateInit;


{Atom: 2D_CameraPositions}

sets;
AtomByName([2D_CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom '2D_CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, s, [2D_CameraPositions], 1, false);
SetAtomSettings([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(45);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
CreateTableColumn(0, 64, [`ID`
1
]);
CreateTableColumn(1, 64, [`Show`
]);
CreateTableColumn(2, 0, [`Name`
]);
CreateTableColumn(3, 0, [`2D3D`
]);
CreateTableColumn(4, 0, [`Visible Height layers`
]);
CreateTableColumn(5, 0, [`Window Width`
]);
CreateTableColumn(6, 0, [`Window Heigth`
]);
CreateTableColumn(7, 0, [`Window Xloc`
]);
CreateTableColumn(8, 0, [`Window Yloc`
]);
CreateTableColumn(9, 0, [`Scale`
]);
CreateTableColumn(10, 0, [`ViewX`
]);
CreateTableColumn(11, 0, [`ViewY`
]);
CreateTableColumn(12, 0, [`ViewZ`
]);
CreateTableColumn(13, 0, [`Roll`
]);
CreateTableColumn(14, 0, [`Pitch`
]);
CreateTableColumn(15, 0, [`Yaw`
]);
SetStatus(0);
ExecLateInit;


{Atom: 3D_CameraPositions}

sets;
AtomByName([3D_CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom '3D_CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [3D_CameraPositions], 1, false);
SetAtomSettings([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(46);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
CreateTableColumn(0, 64, [`ID`
1
]);
CreateTableColumn(1, 64, [`Show`
]);
CreateTableColumn(2, 0, [`Name`
]);
CreateTableColumn(3, 0, [`2D3D`
]);
CreateTableColumn(4, 0, [`Visible Height layers`
]);
CreateTableColumn(5, 0, [`Window Width`
]);
CreateTableColumn(6, 0, [`Window Heigth`
]);
CreateTableColumn(7, 0, [`Window Xloc`
]);
CreateTableColumn(8, 0, [`Window Yloc`
]);
CreateTableColumn(9, 0, [`Scale`
]);
CreateTableColumn(10, 0, [`ViewX`
]);
CreateTableColumn(11, 0, [`ViewY`
]);
CreateTableColumn(12, 0, [`ViewZ`
]);
CreateTableColumn(13, 0, [`Roll`
]);
CreateTableColumn(14, 0, [`Pitch`
]);
CreateTableColumn(15, 0, [`Yaw`
]);
SetStatus(0);
ExecLateInit;
Up;
Up;


{Atom: PD_Environment_DisplayAtom}

sets;
AtomByName([DisplayAtom], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DisplayAtom'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [PD_Environment_DisplayAtom], 1, false);
SetAtomSettings([], 0, 1065984);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Tools.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(47);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: PD_Environment}

sets;
AtomByName([PD_Environment], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Environment'. Inheriting from BaseClass.]));
CreateAtom(a, s, [PD_Environment], 1, false);
SetAtt([minX], -10);
SetAtt([minY], -10);
SetAtt([maxX], 70);
SetAtt([maxY], 70);
SetAtt([NrPortals], 0);
SetAtt([ContainsECMNetwork], 1);
SetAtt([ActivityID], 4);
SetAtt([3DBackgroundColor], 15780518);
SetAtt([2DBackgroundColor], 16777215);
SetAtt([GridSize], 0.1);
SetAtt([ActiveHeightLayer], 1);
SetTextAtt([ShowLocation], [0]);
SetAtt([DisableStitching], 0);
SetAtt([DisablePreProcessing], 0);
SetAtt([IndicativeCorridorID], 0);
SetAtt([ActivityIDDynamic], 4);
SetAtt([MinimumClearance], 0.1);
SetAtt([ECMInformationType], 0);
SetAtt([ShowExcludeFromNetwork], 1);
SetAtt([LayoutChecked], 1);
SetAtt([ShowPhysicalDistancing], 0);
SetAtt([PhysicalDistanceColor], 65280);
SetAtt([PhysicalDistanceViolationColor], 255);
SetAtomSettings([], 0, 551984);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(48);
SetSize(80, 80, 0);
SetTranslation(-10, -10, 0);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 4);
CreateTableColumn(0, 64, [0
1
]);
CreateTableColumn(1, 64, [`Layer`
1
]);
CreateTableColumn(2, 64, [`Rank`
1
]);
CreateTableColumn(3, 64, [`Ref`
872077088
]);
CreateTableColumn(4, 0, [4
]);
CreatePoints(4);
SetPoint(0, 0, 0, 0);
SetPoint(1, 60, 0, 0);
SetPoint(2, 60, 60, 0);
SetPoint(3, 0, 60, 0);
UpdateExtremePoints;
SetStatus(0);
ExecLateInit;


{Atom: HeightLayer_1}

sets;
AtomByName([HeightLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'HeightLayer'. Inheriting from BaseClass.]));
CreateAtom(a, s, [HeightLayer_1], 1, false);
SetAtt([CheckContentInView], 2);
SetAtt([xCenter2D], 30);
SetAtt([yCenter2D], 30);
SetAtt([xRadius2D], 30.3);
SetAtt([yRadius2D], 30.3);
SetAtt([xCenter3D], 30);
SetAtt([yCenter3D], 30);
SetAtt([zCenter3D], 1.5);
SetAtt([Radius3D], 42.8769168667711);
SetAtt([lstWalkableAreas], 0);
SetAtt([lstObstacles], 0);
SetAtt([lstOpenings], 0);
SetAtt([lstUserActions], 0);
SetAtt([********], 370636352);
SetAtt([lstEdges], 0);
SetAtt([lstAgents], 0);
SetAtt([lstActivities], 0);
SetAtt([LayerID], 1);
SetAtt([PrimitiveType], 2);
SetAtt([GL_Type], 7);
SetAtt([Winding], 0);
SetAtt([ECMColor], 16711884);
SetAtt([AllowedPrimitiveTypes], 3);
SetAtt([DrawBackground2D], 0);
SetAtt([DrawBackground3D], 0);
SetAtt([BackgroundScale], 1);
SetAtt([BackgroundXLoc], 0);
SetAtt([BackgroundYLoc], 0);
SetAtt([BackgroundRotate], 0);
SetAtt([Transparency], 0);
SetAtt([DrawInterior], 2);
SetAtt([DrawExterior], 3);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([ColorInterior2D], 0);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([TextureID], 0);
SetAtt([**NOTUSED**], 4402);
SetAtt([HeightDifference], 0);
SetAtt([DrawMultipleContours], 0);
SetAtt([Height], 0);
SetAtt([DrawSides], 7);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([RotationContent], 0);
SetAtt([**NOTUSED2**], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([DefaultVisibility], 3);
SetAtt([lstResources], 0);
SetAtt([lstTransportation], 0);
SetAtt([zRadius3D], 1.5);
SetAtt([AlwaysCreateWalkableArea], 0);
SetExprAtt([ResetCode], [0]);
SetAtt([BackgroundZLoc], 0);
SetAtt([UseBoundingBox], 0);
SetAtomSettings([], 8421504, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(49);
SetSize(60, 60, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 60, 0);
SetPoint(1, 60, 60, 0);
SetPoint(2, 60, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 2);
SetStatus(0);
ExecLateInit;


{Atom: lstWalkableAreas}

sets;
AtomByName([lstWalkableAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstWalkableAreas'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstWalkableAreas], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(50);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: WalkableArea_1}

sets;
AtomByName([WalkableArea], Main);
if(not(AtomExists), Error([Cannot find mother atom 'WalkableArea'. Inheriting from BaseClass.]));
CreateAtom(a, s, [WalkableArea_1], 1, false);
SetAtt([PrimitiveType], 1);
SetAtt([GL_Type], 7);
SetAtt([Radius], 0);
SetAtt([Winding], 0);
SetAtt([AllowedPrimitiveTypes], 31);
SetAtt([Transparency], 0);
SetAtt([Height], 0);
SetAtt([DrawSides], 7);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([TextureID], 0);
SetAtt([DrawInterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetExprAtt([ColorExterior3D], [ColorBlack]);
SetAtt([DrawMultipleContours], 0);
SetAtt([InViewRadius], 42.8506709399048);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([Direction], 3);
SetExprAtt([DirectionOriginal], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([PercSpeedUp], [SlopeSpeed_DetermineHorizontalSpeedPerc(WalkableArea_GetSlope(c), DIRECTION_UP)]);
SetExprAtt([PercSpeedDown], [SlopeSpeed_DetermineHorizontalSpeedPerc(WalkableArea_GetSlope(c), DIRECTION_DOWN)]);
SetAtt([DiscomfortFactorUp], 0);
SetAtt([DiscomfortFactorDown], 0);
SetAtt([BlockSides], 3);
SetAtt([SplitEdgeID1], -1);
SetAtt([SplitEdgeID2], -1);
SetAtt([CreateSplitEdges], 0);
SetAtt([Token], 0);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtomSettings([], 16777215, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([WalkableAreas]));
SetChannelRanges(0, 999, 0, 999);
SetLoadAtomID(51);
SetLoc(30, 30, 0);
SetSize(60.6, 60.6, 0);
SetTranslation(-30.3, -30.3, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 60.6, 0);
SetPoint(1, 60.6, 60.6, 0);
SetPoint(2, 60.6, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstObstacles}

sets;
AtomByName([lstObstacles], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstObstacles'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstObstacles], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(52);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: Obstacle_2}

sets;
AtomByName([Obstacle], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Obstacle'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Obstacle_2], 1, false);
SetAtt([PrimitiveType], 2);
SetAtt([GL_Type], 7);
SetAtt([Radius], 0);
SetAtt([Winding], 0);
SetAtt([AllowedPrimitiveTypes], 127);
SetAtt([Transparency], 0);
SetAtt([Height], 1);
SetAtt([DrawSides], 7);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([TextureID], 0);
SetAtt([DrawInterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetAtt([DrawExterior], 1);
SetExprAtt([ColorExterior2D], [ColorBlack]);
SetExprAtt([ColorExterior3D], [ColorBlack]);
SetAtt([InviewRadius], 17.6776695296637);
SetAtt([DrawMultipleContours], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([WallWidth], 0.2);
SetAtt([CloseLoop], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([Model3DSettingsIndex], 0);
SetAtt([Subtype], 0);
SetAtomSettings([], 10789024, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Obstacles]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(53);
SetLoc(47.5, 12.5, 0);
SetSize(25, 25, 0);
SetTranslation(-12.5, -12.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 25, 0);
SetPoint(1, 25, 25, 0);
SetPoint(2, 25, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(0);
ExecLateInit;


{Atom: Obstacle_3}

sets;
AtomByName([Obstacle], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Obstacle'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Obstacle_3], 1, false);
SetAtt([PrimitiveType], 2);
SetAtt([GL_Type], 7);
SetAtt([Radius], 0);
SetAtt([Winding], 0);
SetAtt([AllowedPrimitiveTypes], 127);
SetAtt([Transparency], 0);
SetAtt([Height], 1);
SetAtt([DrawSides], 7);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([TextureID], 0);
SetAtt([DrawInterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetAtt([DrawExterior], 1);
SetExprAtt([ColorExterior2D], [ColorBlack]);
SetExprAtt([ColorExterior3D], [ColorBlack]);
SetAtt([InviewRadius], 17.6776695296637);
SetAtt([DrawMultipleContours], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([WallWidth], 0.2);
SetAtt([CloseLoop], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([Model3DSettingsIndex], 0);
SetAtt([Subtype], 0);
SetAtomSettings([], 10789024, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Obstacles]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(54);
SetLoc(12.5, 47.5, 0);
SetSize(25, 25, 0);
SetTranslation(-12.5, -12.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 25, 0);
SetPoint(1, 25, 25, 0);
SetPoint(2, 25, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(0);
ExecLateInit;


{Atom: Obstacle_4}

sets;
AtomByName([Obstacle], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Obstacle'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Obstacle_4], 1, false);
SetAtt([PrimitiveType], 2);
SetAtt([GL_Type], 7);
SetAtt([Radius], 0);
SetAtt([Winding], 0);
SetAtt([AllowedPrimitiveTypes], 127);
SetAtt([Transparency], 0);
SetAtt([Height], 1);
SetAtt([DrawSides], 7);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([TextureID], 0);
SetAtt([DrawInterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetAtt([DrawExterior], 1);
SetExprAtt([ColorExterior2D], [ColorBlack]);
SetExprAtt([ColorExterior3D], [ColorBlack]);
SetAtt([InviewRadius], 17.6776695296637);
SetAtt([DrawMultipleContours], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([WallWidth], 0.2);
SetAtt([CloseLoop], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([Model3DSettingsIndex], 0);
SetAtt([Subtype], 0);
SetAtomSettings([], 10789024, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Obstacles]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(55);
SetLoc(47.5, 47.5, 0);
SetSize(25, 25, 0);
SetTranslation(-12.5, -12.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 25, 0);
SetPoint(1, 25, 25, 0);
SetPoint(2, 25, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(0);
ExecLateInit;


{Atom: Obstacle_4}

sets;
AtomByName([Obstacle], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Obstacle'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Obstacle_4], 1, false);
SetAtt([PrimitiveType], 1);
SetAtt([GL_Type], 7);
SetAtt([Radius], 0);
SetAtt([Winding], 0);
SetAtt([AllowedPrimitiveTypes], 127);
SetAtt([Transparency], 0);
SetAtt([Height], 3);
SetAtt([DrawSides], 7);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([TextureID], 0);
SetAtt([DrawInterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetAtt([DrawExterior], 1);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetExprAtt([ColorExterior3D], [ColorBlack]);
SetAtt([InviewRadius], 17.6707385244647);
SetAtt([DrawMultipleContours], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([WallWidth], 0);
SetAtt([CloseLoop], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([Model3DSettingsIndex], 0);
SetAtt([Subtype], 0);
SetAtomSettings([], 10789024, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Obstacles]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(56);
SetLoc(12.35, 12.55, 0);
SetSize(24.9, 24.9, 3);
SetTranslation(-12.45, -12.45, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
CreatePoints(4);
SetPoint(0, 0, 24.9, 0);
SetPoint(1, 24.9, 24.9, 0);
SetPoint(2, 24.9, 0, 0);
SetPoint(3, 0, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstOpenings}

sets;
AtomByName([lstOpenings], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstOpenings'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstOpenings], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(57);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstTransportation}

sets;
AtomByName([lstTransportation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTransportation], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(58);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstActivities}

sets;
AtomByName([lstActivities], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstActivities'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstActivities], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(59);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: EntryExit_1}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EntryExit_1], 1, false);
SetAtt([ID], 1);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 2);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetAtt([RepetitiveMode], 2001);
SetExprAtt([RepeatTime], [0]);
SetExprAtt([OffsetTime], [0]);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [4DS[{**Random point on a specific line**}
do(  
  var([valIndex], vbValue, 2),    {**Line index**}
  var([x1], vbValue),
  var([y1], vbValue),
  var([x2], vbValue),
  var([y2], vbValue),
  
  Points_Get(c, valIndex),  
  x1 := vx,
  y1 := vy,
  
  Points_Get(c, mod(valIndex, Points_Count(c)) + 1),
  x2 := vx,
  y2 := vy,
  
  getLineLerp(    
    x1, y1, x2, y2,
    (Agent_GetSidePreference(i) + 1) / 2 {**Interpolation control factor**}
  ),
  
  Primitives_GetAbsoluteLocationOfPoint(c, PDEnvAtom, vx, vy)
)]4DS]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetAtt([ColorInterior2D], 16711680);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetExprAtt([ServiceTime], [0]);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 0);
SetAtt([DrawingDirection], 0);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetExprAtt([QueueStrategy], [1]);
SetExprAtt([QueueDiscipline], [0]);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetExprAtt([LocationDeviation], [0]);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}
do(
  var([valIndex], vbValue, 1 {indicative corridor index}),
  if(
    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),
    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})
  )
)]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
SetAtomSettings([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(60);
SetLoc(2.5, 30, 0);
SetSize(5, 10, 1);
SetTranslation(-2.5, -5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
CreateTableColumn(0, 64, [``
1
]);
CreateTableColumn(1, 64, [`Start time`
-1
]);
CreateTableColumn(2, 76, [`Time open`
0
]);
CreateTableColumn(3, 93, [`Trigger on start`
0
]);
CreateTableColumn(4, 115, [`Trigger on end`
0
]);
CreatePoints(4);
SetPoint(0, 0, 0, 0);
SetPoint(1, 5, 0, 0);
SetPoint(2, 5, 10, 0);
SetPoint(3, 0, 10, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
ExecLateInit;


{Atom: EntryExit_2}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_2], 1, false);
SetAtt([ID], 2);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 2);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetAtt([RepetitiveMode], 2001);
SetExprAtt([RepeatTime], [0]);
SetExprAtt([OffsetTime], [0]);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [4DS[{**Random point on a specific line**}
do(  
  var([valIndex], vbValue, 3),    {**Line index**}
  var([x1], vbValue),
  var([y1], vbValue),
  var([x2], vbValue),
  var([y2], vbValue),
  
  Points_Get(c, valIndex),  
  x1 := vx,
  y1 := vy,
  
  Points_Get(c, mod(valIndex, Points_Count(c)) + 1),
  x2 := vx,
  y2 := vy,
  
  getLineLerp(    
    x1, y1, x2, y2,
    (Agent_GetSidePreference(i) + 1) / 2 {**Interpolation control factor**}
  ),
  
  Primitives_GetAbsoluteLocationOfPoint(c, PDEnvAtom, vx, vy)
)]4DS]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [{**Set color of the agent**} Color(i) := ColorFuchsia]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetAtt([ColorInterior2D], 16711680);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetExprAtt([ServiceTime], [0]);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 0);
SetAtt([DrawingDirection], 0);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetExprAtt([QueueStrategy], [1]);
SetExprAtt([QueueDiscipline], [0]);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetExprAtt([LocationDeviation], [0]);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}
do(
  var([valIndex], vbValue, 1 {indicative corridor index}),
  if(
    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),
    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})
  )
)]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
SetAtomSettings([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(61);
SetLoc(30, 2.5, 0);
SetSize(10, 5, 1);
SetTranslation(-5, -2.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
CreateTableColumn(0, 64, [``
1
]);
CreateTableColumn(1, 64, [`Start time`
-1
]);
CreateTableColumn(2, 76, [`Time open`
0
]);
CreateTableColumn(3, 93, [`Trigger on start`
0
]);
CreateTableColumn(4, 115, [`Trigger on end`
0
]);
CreatePoints(4);
SetPoint(0, 0, 0, 0);
SetPoint(1, 10, 0, 0);
SetPoint(2, 10, 5, 0);
SetPoint(3, 0, 5, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
ExecLateInit;


{Atom: EntryExit_3}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_3], 1, false);
SetAtt([ID], 3);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 2);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetAtt([RepetitiveMode], 2001);
SetExprAtt([RepeatTime], [0]);
SetExprAtt([OffsetTime], [0]);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [4DS[{**Random point on a specific line**}
do(  
  var([valIndex], vbValue, 4),    {**Line index**}
  var([x1], vbValue),
  var([y1], vbValue),
  var([x2], vbValue),
  var([y2], vbValue),
  
  Points_Get(c, valIndex),  
  x1 := vx,
  y1 := vy,
  
  Points_Get(c, mod(valIndex, Points_Count(c)) + 1),
  x2 := vx,
  y2 := vy,
  
  getLineLerp(    
    x1, y1, x2, y2,
    (Agent_GetSidePreference(i) + 1) / 2 {**Interpolation control factor**}
  ),
  
  Primitives_GetAbsoluteLocationOfPoint(c, PDEnvAtom, vx, vy)
)]4DS]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [{**Set color of the agent**} Color(i) := ColorRed]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetAtt([ColorInterior2D], 16711680);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetExprAtt([ServiceTime], [0]);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 0);
SetAtt([DrawingDirection], 0);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetExprAtt([QueueStrategy], [1]);
SetExprAtt([QueueDiscipline], [0]);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetExprAtt([LocationDeviation], [0]);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}
do(
  var([valIndex], vbValue, 1 {indicative corridor index}),
  if(
    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),
    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})
  )
)]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
SetAtomSettings([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(62);
SetLoc(57.5, 30, 0);
SetSize(5, 10, 1);
SetTranslation(-2.5, -5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
CreateTableColumn(0, 64, [``
1
]);
CreateTableColumn(1, 64, [`Start time`
-1
]);
CreateTableColumn(2, 76, [`Time open`
0
]);
CreateTableColumn(3, 93, [`Trigger on start`
0
]);
CreateTableColumn(4, 115, [`Trigger on end`
0
]);
CreatePoints(4);
SetPoint(0, 0, 0, 0);
SetPoint(1, 5, 0, 0);
SetPoint(2, 5, 10, 0);
SetPoint(3, 0, 10, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
ExecLateInit;


{Atom: EntryExit_4}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_4], 1, false);
SetAtt([ID], 4);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 2);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetAtt([RepetitiveMode], 2001);
SetExprAtt([RepeatTime], [0]);
SetExprAtt([OffsetTime], [0]);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [4DS[{**Random point on a specific line**}
do(  
  var([valIndex], vbValue, 1),    {**Line index**}
  var([x1], vbValue),
  var([y1], vbValue),
  var([x2], vbValue),
  var([y2], vbValue),
  
  Points_Get(c, valIndex),  
  x1 := vx,
  y1 := vy,
  
  Points_Get(c, mod(valIndex, Points_Count(c)) + 1),
  x2 := vx,
  y2 := vy,
  
  getLineLerp(    
    x1, y1, x2, y2,
    (Agent_GetSidePreference(i) + 1) / 2 {**Interpolation control factor**}
  ),
  
  Primitives_GetAbsoluteLocationOfPoint(c, PDEnvAtom, vx, vy)
)]4DS]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [{**Set color of the agent**} Color(i) := ColorLime]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetAtt([ColorInterior2D], 16711680);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetExprAtt([ServiceTime], [0]);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 0);
SetAtt([DrawingDirection], 0);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetExprAtt([QueueStrategy], [1]);
SetExprAtt([QueueDiscipline], [0]);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetExprAtt([LocationDeviation], [0]);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}
do(
  var([valIndex], vbValue, 1 {indicative corridor index}),
  if(
    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),
    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})
  )
)]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
SetAtomSettings([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\atom.ico]));
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
SetLoadAtomID(63);
SetLoc(30, 57.5, 0);
SetSize(10, 5, 1);
SetTranslation(-5, -2.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
CreateTableColumn(0, 64, [``
1
]);
CreateTableColumn(1, 64, [`Start time`
-1
]);
CreateTableColumn(2, 76, [`Time open`
0
]);
CreateTableColumn(3, 93, [`Trigger on start`
0
]);
CreateTableColumn(4, 115, [`Trigger on end`
0
]);
CreatePoints(4);
SetPoint(0, 0, 0, 0);
SetPoint(1, 10, 0, 0);
SetPoint(2, 10, 5, 0);
SetPoint(3, 0, 5, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
ExecLateInit;
Up;


{Atom: lstUserActions}

sets;
AtomByName([lstUserActions], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstUserActions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstUserActions], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(64);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstAgents}

sets;
AtomByName([lstAgents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstAgents'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstAgents], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(65);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstResources}

sets;
AtomByName([lstResources], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstResources'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstResources], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(66);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: ECM_Visualization}

sets;
AtomByName([ECM_Visualization], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ECM_Visualization'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ECM_Visualization], 1, false);
SetAtomSettings([], 0, 542768);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([ECMNetwork]));
SetLoadAtomID(67);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;
Up;


{Atom: AgentDrawInformation}

sets;
AtomByName([AgentDrawInformation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentDrawInformation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentDrawInformation], 1, false);
SetAtt([InformationType2D], 14);
SetAtt([InformationType3D], 0);
SetAtomSettings([], 0, 546864);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(68);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: RunClock}

sets;
AtomByName([RunClock], Main);
if(not(AtomExists), Error([Cannot find mother atom 'RunClock'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [RunClock], 1, false);
SetExprAtt([text], [4DS[DateTime(TimeFromNow(att([startTime], c)), [hh:mm:ss])]4DS]);
SetExprAtt([textcolor], [1]);
SetExprAtt([backgroundcolor], [colorwhite]);
SetAtt([fontsize], 3);
SetExprAtt([font], [4DS[[Calibri]]4DS]);
SetAtt([fontscale], 0);
SetAtt([italic], 0);
SetAtt([pitch], 0);
SetAtt([xpitch], -45);
SetAtt([zpitch], 0);
SetAtt([3DEnabled], 0);
SetAtt([FontSize3D], 0.2);
SetAtt([2DEnabled], 0);
SetAtt([StartTime], 0);
SetAtt([EvacuationTime], 0);
SetAtt([OffSet3D], 0);
SetAtt([OffSet2D], 0);
SetAtt([Show], 3);
SetAtomSettings([], 15, 542768);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Atom.ico]));
Layer(LayerByName([PD_Environment]));
SetChannels(1, 0);
SetChannelRanges(1, 1, 0, 0);
SetLoadAtomID(69);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: MovieRecord}

sets;
AtomByName([MovieRecord], Main);
if(not(AtomExists), Error([Cannot find mother atom 'MovieRecord'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [MovieRecord], 1, false);
SetExprAtt([Filename], [4DS[WorkDir(Concat(ExtractPreName(TextAtt(1, Model)), [.avi]))]4DS]);
SetAtt([Framerate], 10);
SetAtt([ActiveRecord], -1);
SetAtt([RecordPaused], 0);
SetAtt([Interval], 0.1);
SetAtt([FromGuiInstance], 0);
SetAtt([Width], 1600);
SetAtt([Height], 900);
SetExprAtt([RecordTime], [0]);
SetAtt([StopRecording], 1);
SetAtomSettings([], 15, 542720);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Atom.ico]));
Layer(LayerByName([DoNotDraw]));
SetChannels(1, 0);
SetChannelRanges(1, 1, 0, 0);
SetLoadAtomID(70);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: SelectionPolygon}

sets;
AtomByName([SelectionPolygon], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SelectionPolygon'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SelectionPolygon], 1, false);
SetAtt([PrimitiveType], 8);
SetAtt([GL_Type], 0);
SetAtt([Winding], 0);
SetAtomSettings([], 0, 528432);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(71);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
Set(Points_PrimitiveType(a), 10);
SetStatus(0);
ExecLateInit;


{Atom: StaticAnalysis}

sets;
AtomByName([StaticAnalysis], Main);
if(not(AtomExists), Error([Cannot find mother atom 'StaticAnalysis'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [StaticAnalysis], 1, false);
SetAtt([NrOfIterations], 500);
SetAtt([Seperation], 0);
SetMultiVectorCompareByFields(1, 0, 1);
SetAtomSettings([], 0, 543922);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\ModelLayout\Ruler.ico]));
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(72);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: PD_Output}

sets;
AtomByName([PD_Output], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Output'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [PD_Output], 1, false);
SetExprAtt([StartTime], [0]);
SetExprAtt([EndTime], [0]);
SetAtt([ColorMapType], 0);
SetAtt([ShowLegend], 1);
SetAtt([LegendSizeX], 2.5);
SetAtt([ActivityRouteHide], 0);
SetAtt([OutputLoaded], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([ColormapRef], 0);
SetTextAtt([ColormapTitle], [0]);
SetAtt([GridSize], 0.2);
SetAtt([ScaleLegend], 1);
SetAtt([TextSize], 1);
SetAtt([GridDensityRadius], 0.6);
SetAtt([SocialCostValue], 8.38);
SetTextAtt([SocialCostCurrency], [€]);
SetAtt([SocialCostDensityFactor], 0.6667);
SetAtt([SocialCostDensityMin], 0.5);
SetAtt([SocialCostDensityMax], 2);
SetAtt([SocialCostSet], 0);
SetAtt([DensitymapShowLayers], 0);
SetDataContainerItem(0, 12, [`Walking`;2;0.5;
`Walking`;2;0.5;
`Walking`;2;0.5;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
`Escalator`;1.5;0;
`Escalator`;1.5;0;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
`Stands`;2;0.5;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
]);
SetMultiVectorCompareByFields(0, 0);
SetMultiVectorCompareByFields(1, 0);
SetAtomSettings([], 0, 543792);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
SetLoadAtomID(73);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: OutputLayer_1}

sets;
AtomByName([OutputLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'OutputLayer'. Inheriting from BaseClass.]));
CreateAtom(a, s, [OutputLayer_1], 1, false);
SetAtt([lstFlowCounters], 0);
SetAtt([lstDensityAreas], 0);
SetAtt([lstColorMaps], 0);
SetAtt([LayerID], 1);
SetAtomSettings([], 8421504, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannels(1, 0);
SetChannelRanges(1, 255, 0, 0);
SetLoadAtomID(74);
SetSize(60, 60, 0);
LockPosition(true);
LockSize(true);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstFlowCounters}

sets;
AtomByName([lstFlowCounters], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstFlowCounters'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstFlowCounters], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(75);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: FlowCounter 1}

sets;
AtomByName([FlowCounter], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FlowCounter'. Inheriting from BaseClass.]));
CreateAtom(a, s, [FlowCounter 1], 1, false);
SetAtt([PrimitiveType], 32);
SetAtt([GL_Type], 1);
SetAtt([AllowedPrimitiveTypes], 32);
SetAtt([FlowCalculated], 0);
SetExprAtt([Color_FlowLeft], [RgbColor(68, 102, 163)]);
SetExprAtt([Color_FlowRight], [RgbColor(243, 156, 53)]);
SetAtt([FlowLeftColumn], 0);
SetAtt([FlowRightColumn], 0);
SetExprAtt([GraphType], [FrequencyNorms_GetDefaultGraphType(atmFrequencyNorms)]);
SetExprAtt([Graph3D], [FrequencyNorms_GetDefaultGraphView3D(atmFrequencyNorms)]);
SetExprAtt([BottomLabelsAngle], [FrequencyNorms_GetBottomLabelsAngle(atmFrequencyNorms)]);
SetExprAtt([MinY], [FrequencyNorms_GetMinY(atmFrequencyNorms)]);
SetExprAtt([MaxY], [FrequencyNorms_GetMaxY(atmFrequencyNorms)]);
SetAtt([Winding], 0);
SetExprAtt([ShowPerMeter], [FrequencyNorms_GetShowPerMeter(atmFrequencyNorms)]);
SetTextAtt([FlowCounterSettings], []);
SetTextAtt([TitleFlowLeft], [Left to Right]);
SetTextAtt([TitleFlowRight], [Right to Left]);
SetAtomSettings([], 0, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\OutputLayout\draw_flowcounter16.ico]));
Layer(LayerByName([Output]));
SetChannelRanges(0, 999, 0, 999);
SetLoadAtomID(76);
SetLoc(8.25, 30, 0.1);
SetSize(10.0005, 0, 0);
SetTranslation(-5.00025, 0, 0);
Set(RotationAs, 89.4270613023165);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(181, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
]);
CreateTableColumn(1, 0, [1
0
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150
160
170
180
190
200
210
220
230
240
250
260
270
280
290
300
310
320
330
340
350
360
370
380
390
400
410
420
430
440
450
460
470
480
490
500
510
520
530
540
550
560
570
580
590
600
610
620
630
640
650
660
670
680
690
700
710
720
730
740
750
760
770
780
790
800
810
820
830
840
850
860
870
880
890
900
910
920
930
940
950
960
970
980
990
1000
1010
1020
1030
1040
1050
1060
1070
1080
1090
1100
1110
1120
1130
1140
1150
1160
1170
1180
1190
1200
1210
1220
1230
1240
1250
1260
1270
1280
1290
1300
1310
1320
1330
1340
1350
1360
1370
1380
1390
1400
1410
1420
1430
1440
1450
1460
1470
1480
1490
1500
1510
1520
1530
1540
1550
1560
1570
1580
1590
1600
1610
1620
1630
1640
1650
1660
1670
1680
1690
1700
1710
1720
1730
1740
1750
1760
1770
1780
1790
1800
]);
CreateTableColumn(2, 0, [2
0
54
80
70
69
64
76
63
70
71
62
63
69
63
71
63
67
67
60
55
55
60
63
66
68
69
79
59
63
68
63
70
67
67
67
68
61
58
63
70
62
51
74
68
71
71
64
71
48
72
59
56
61
68
67
44
59
61
65
67
59
72
63
76
69
69
70
55
73
78
68
71
63
66
68
74
77
50
85
63
53
60
68
59
69
66
69
65
60
80
68
75
64
62
69
76
80
64
78
52
71
72
78
63
72
58
59
63
63
75
66
73
61
73
60
80
68
62
75
58
66
62
60
67
65
72
66
73
62
59
74
66
62
60
62
68
71
65
70
45
68
58
68
66
75
51
63
69
75
56
66
69
72
74
73
64
91
71
55
71
70
57
61
58
64
66
64
66
62
65
70
83
65
59
64
72
77
76
78
67
84
]);
CreateTableColumn(3, 0, [3
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreatePoints(2);
SetPoint(0, 0, 0, 0);
SetPoint(1, 10.0005, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 1);
SetStatus(0);
ExecLateInit;


{Atom: FlowCounter 2}

sets;
AtomByName([FlowCounter], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FlowCounter'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [FlowCounter 2], 1, false);
SetAtt([PrimitiveType], 32);
SetAtt([GL_Type], 1);
SetAtt([AllowedPrimitiveTypes], 32);
SetAtt([FlowCalculated], 0);
SetExprAtt([Color_FlowLeft], [RgbColor(68, 102, 163)]);
SetExprAtt([Color_FlowRight], [RgbColor(243, 156, 53)]);
SetAtt([FlowLeftColumn], 0);
SetAtt([FlowRightColumn], 0);
SetExprAtt([GraphType], [FrequencyNorms_GetDefaultGraphType(atmFrequencyNorms)]);
SetExprAtt([Graph3D], [FrequencyNorms_GetDefaultGraphView3D(atmFrequencyNorms)]);
SetExprAtt([BottomLabelsAngle], [FrequencyNorms_GetBottomLabelsAngle(atmFrequencyNorms)]);
SetExprAtt([MinY], [FrequencyNorms_GetMinY(atmFrequencyNorms)]);
SetExprAtt([MaxY], [FrequencyNorms_GetMaxY(atmFrequencyNorms)]);
SetAtt([Winding], 0);
SetExprAtt([ShowPerMeter], [FrequencyNorms_GetShowPerMeter(atmFrequencyNorms)]);
SetTextAtt([FlowCounterSettings], []);
SetTextAtt([TitleFlowLeft], [Left to Right]);
SetTextAtt([TitleFlowRight], [Right to Left]);
SetAtomSettings([], 0, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\OutputLayout\draw_flowcounter16.ico]));
Layer(LayerByName([Output]));
SetChannelRanges(0, 999, 0, 999);
SetLoadAtomID(77);
SetLoc(29.85, 8, 0.1);
SetSize(10.1, 0, 0);
SetTranslation(-5.05, 0, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(181, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
]);
CreateTableColumn(1, 0, [1
0
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150
160
170
180
190
200
210
220
230
240
250
260
270
280
290
300
310
320
330
340
350
360
370
380
390
400
410
420
430
440
450
460
470
480
490
500
510
520
530
540
550
560
570
580
590
600
610
620
630
640
650
660
670
680
690
700
710
720
730
740
750
760
770
780
790
800
810
820
830
840
850
860
870
880
890
900
910
920
930
940
950
960
970
980
990
1000
1010
1020
1030
1040
1050
1060
1070
1080
1090
1100
1110
1120
1130
1140
1150
1160
1170
1180
1190
1200
1210
1220
1230
1240
1250
1260
1270
1280
1290
1300
1310
1320
1330
1340
1350
1360
1370
1380
1390
1400
1410
1420
1430
1440
1450
1460
1470
1480
1490
1500
1510
1520
1530
1540
1550
1560
1570
1580
1590
1600
1610
1620
1630
1640
1650
1660
1670
1680
1690
1700
1710
1720
1730
1740
1750
1760
1770
1780
1790
1800
]);
CreateTableColumn(2, 0, [2
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreateTableColumn(3, 0, [3
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreatePoints(2);
SetPoint(0, 0, 0, 0);
SetPoint(1, 10.1, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 1);
SetStatus(0);
ExecLateInit;


{Atom: FlowCounter 3}

sets;
AtomByName([FlowCounter], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FlowCounter'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [FlowCounter 3], 1, false);
SetAtt([PrimitiveType], 32);
SetAtt([GL_Type], 1);
SetAtt([AllowedPrimitiveTypes], 32);
SetAtt([FlowCalculated], 0);
SetExprAtt([Color_FlowLeft], [RgbColor(68, 102, 163)]);
SetExprAtt([Color_FlowRight], [RgbColor(243, 156, 53)]);
SetAtt([FlowLeftColumn], 0);
SetAtt([FlowRightColumn], 0);
SetExprAtt([GraphType], [FrequencyNorms_GetDefaultGraphType(atmFrequencyNorms)]);
SetExprAtt([Graph3D], [FrequencyNorms_GetDefaultGraphView3D(atmFrequencyNorms)]);
SetExprAtt([BottomLabelsAngle], [FrequencyNorms_GetBottomLabelsAngle(atmFrequencyNorms)]);
SetExprAtt([MinY], [FrequencyNorms_GetMinY(atmFrequencyNorms)]);
SetExprAtt([MaxY], [FrequencyNorms_GetMaxY(atmFrequencyNorms)]);
SetAtt([Winding], 0);
SetExprAtt([ShowPerMeter], [FrequencyNorms_GetShowPerMeter(atmFrequencyNorms)]);
SetTextAtt([FlowCounterSettings], []);
SetTextAtt([TitleFlowLeft], [Left to Right]);
SetTextAtt([TitleFlowRight], [Right to Left]);
SetAtomSettings([], 0, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\OutputLayout\draw_flowcounter16.ico]));
Layer(LayerByName([Output]));
SetChannelRanges(0, 999, 0, 999);
SetLoadAtomID(78);
SetLoc(51.3, 29.9, 0.1);
SetSize(9.900505, 0, 0);
SetTranslation(-4.950253, 0, 0);
Set(RotationAs, 89.4212744343922);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(181, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
]);
CreateTableColumn(1, 0, [1
0
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150
160
170
180
190
200
210
220
230
240
250
260
270
280
290
300
310
320
330
340
350
360
370
380
390
400
410
420
430
440
450
460
470
480
490
500
510
520
530
540
550
560
570
580
590
600
610
620
630
640
650
660
670
680
690
700
710
720
730
740
750
760
770
780
790
800
810
820
830
840
850
860
870
880
890
900
910
920
930
940
950
960
970
980
990
1000
1010
1020
1030
1040
1050
1060
1070
1080
1090
1100
1110
1120
1130
1140
1150
1160
1170
1180
1190
1200
1210
1220
1230
1240
1250
1260
1270
1280
1290
1300
1310
1320
1330
1340
1350
1360
1370
1380
1390
1400
1410
1420
1430
1440
1450
1460
1470
1480
1490
1500
1510
1520
1530
1540
1550
1560
1570
1580
1590
1600
1610
1620
1630
1640
1650
1660
1670
1680
1690
1700
1710
1720
1730
1740
1750
1760
1770
1780
1790
1800
]);
CreateTableColumn(2, 0, [2
0
0
0
1
25
46
78
74
67
66
73
63
68
76
62
69
66
70
66
60
64
63
67
65
47
55
75
60
64
68
71
69
68
64
61
69
58
77
71
57
60
67
73
65
60
56
68
61
75
69
77
55
61
62
60
66
57
67
62
46
57
65
59
67
67
59
72
69
63
79
67
75
64
67
71
77
55
72
72
70
77
62
67
61
56
62
66
63
58
76
70
57
66
79
75
62
79
66
60
64
78
76
72
63
61
67
68
82
73
70
55
62
65
69
63
65
62
77
62
75
67
67
59
80
73
49
64
57
69
69
65
82
63
61
80
52
70
57
70
54
75
65
66
56
66
67
62
64
58
59
63
68
71
53
70
78
81
68
74
66
72
75
76
65
56
61
65
67
55
66
63
67
59
58
76
72
66
70
71
69
68
]);
CreateTableColumn(3, 0, [3
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreatePoints(2);
SetPoint(0, 0, 0, 0);
SetPoint(1, 9.9005, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 1);
SetStatus(0);
ExecLateInit;


{Atom: FlowCounter 4}

sets;
AtomByName([FlowCounter], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FlowCounter'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [FlowCounter 4], 1, false);
SetAtt([PrimitiveType], 32);
SetAtt([GL_Type], 1);
SetAtt([AllowedPrimitiveTypes], 32);
SetAtt([FlowCalculated], 0);
SetExprAtt([Color_FlowLeft], [RgbColor(68, 102, 163)]);
SetExprAtt([Color_FlowRight], [RgbColor(243, 156, 53)]);
SetAtt([FlowLeftColumn], 0);
SetAtt([FlowRightColumn], 0);
SetExprAtt([GraphType], [FrequencyNorms_GetDefaultGraphType(atmFrequencyNorms)]);
SetExprAtt([Graph3D], [FrequencyNorms_GetDefaultGraphView3D(atmFrequencyNorms)]);
SetExprAtt([BottomLabelsAngle], [FrequencyNorms_GetBottomLabelsAngle(atmFrequencyNorms)]);
SetExprAtt([MinY], [FrequencyNorms_GetMinY(atmFrequencyNorms)]);
SetExprAtt([MaxY], [FrequencyNorms_GetMaxY(atmFrequencyNorms)]);
SetAtt([Winding], 0);
SetExprAtt([ShowPerMeter], [FrequencyNorms_GetShowPerMeter(atmFrequencyNorms)]);
SetTextAtt([FlowCounterSettings], []);
SetTextAtt([TitleFlowLeft], [Left to Right]);
SetTextAtt([TitleFlowRight], [Right to Left]);
SetAtomSettings([], 0, 8240);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\OutputLayout\draw_flowcounter16.ico]));
Layer(LayerByName([Output]));
SetChannelRanges(0, 999, 0, 999);
SetLoadAtomID(79);
SetLoc(30, 51.3, 0.1);
SetSize(10, 0, 0);
SetTranslation(-5, 0, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(181, 3);
CreateTableColumn(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
]);
CreateTableColumn(1, 0, [1
0
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150
160
170
180
190
200
210
220
230
240
250
260
270
280
290
300
310
320
330
340
350
360
370
380
390
400
410
420
430
440
450
460
470
480
490
500
510
520
530
540
550
560
570
580
590
600
610
620
630
640
650
660
670
680
690
700
710
720
730
740
750
760
770
780
790
800
810
820
830
840
850
860
870
880
890
900
910
920
930
940
950
960
970
980
990
1000
1010
1020
1030
1040
1050
1060
1070
1080
1090
1100
1110
1120
1130
1140
1150
1160
1170
1180
1190
1200
1210
1220
1230
1240
1250
1260
1270
1280
1290
1300
1310
1320
1330
1340
1350
1360
1370
1380
1390
1400
1410
1420
1430
1440
1450
1460
1470
1480
1490
1500
1510
1520
1530
1540
1550
1560
1570
1580
1590
1600
1610
1620
1630
1640
1650
1660
1670
1680
1690
1700
1710
1720
1730
1740
1750
1760
1770
1780
1790
1800
]);
CreateTableColumn(2, 0, [2
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreateTableColumn(3, 0, [3
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
CreatePoints(2);
SetPoint(0, 0, 0, 0);
SetPoint(1, 10, 0, 0);
UpdateExtremePoints;
Set(Points_PrimitiveType(a), 1);
SetStatus(0);
ExecLateInit;
Up;


{Atom: lstDensityAreas}

sets;
AtomByName([lstDensityAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstDensityAreas'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstDensityAreas], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(80);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstColorMaps}

sets;
AtomByName([lstColorMaps], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstColorMaps'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstColorMaps], 1, false);
SetAtomSettings([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(81);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: OutputSettings}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [OutputSettings], 1, false);
SetAtomSettings([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(82);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: lstNorms}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstNorms], 1, false);
SetAtomSettings([], 0, 1024);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
SetLoadAtomID(83);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: DensityNorms}

sets;
AtomByName([DensityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, s, [DensityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 10);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(84);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
7
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
0.308
0.431
0.718
1.076
2.153
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
0.308
0.431
0.718
1.076
2.153
3
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`few`
`few/mediocre`
`mediocre`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
ExecLateInit;


{Atom: DensityNorms_Walkways}

sets;
AtomByName([DensityNorms_Walkways], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms_Walkways'. Inheriting from BaseClass.]));
CreateAtom(a, s, [DensityNorms_Walkways], 1, false);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(85);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
7
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
0.308
0.431
0.718
1.076
2.153
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
0.308
0.431
0.718
1.076
2.153
3
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`few`
`few/mediocre`
`mediocre`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
ExecLateInit;


{Atom: DensityNorms_Stairways}

sets;
AtomByName([DensityNorms_Stairways], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms_Stairways'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [DensityNorms_Stairways], 1, false);
SetAtomSettings([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(86);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
7
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
0.538
0.718
1.076
1.538
2.691
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
0.538
0.718
1.076
1.538
2.691
3
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`few`
`few/mediocre`
`mediocre`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
ExecLateInit;


{Atom: DensityTimeNorms}

sets;
AtomByName([DensityTimeNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityTimeNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [DensityTimeNorms], 1, false);
SetAtomSettings([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(87);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
7
]);
CreateTableColumn(1, 64, [`Level`
0
1
2
3
4
5
6
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
60
120
180
240
300
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
60
120
180
240
300
301
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
CreateTableColumn(5, 64, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: FrequencyNorms}

sets;
AtomByName([FrequencyNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FrequencyNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [FrequencyNorms], 1, false);
SetAtt([AutoRescaleNorms], 17002);
SetAtt([BlendColors], 0);
SetAtt([Interval], 10);
SetAtt([IntervalsBetweenLabels], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([DefaultGraphType], 3);
SetAtt([DefaultGraphView3D], 0);
SetAtt([MinY], 0);
SetAtt([MaxY], 0);
SetAtt([ShowPerMeter], 0);
SetAtt([UserDefinedColormaps], 0);
SetAtt([StatisticType], 18001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(88);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`Level`
0
1
2
3
4
5
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
200
400
600
800
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
200
400
600
800
801
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
]);
CreateTableColumn(5, 105, [`Description`
``
`A`
`B`
`C`
`D`
`E`
]);
SetStatus(0);
ExecLateInit;


{Atom: TimeOccupiedNorms}

sets;
AtomByName([TimeOccupiedNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TimeOccupiedNorms'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TimeOccupiedNorms], 1, false);
SetAtomSettings([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(89);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
7
]);
CreateTableColumn(1, 64, [`Level`
0
1
2
3
4
5
6
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
60
120
180
240
300
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
60
120
180
240
300
301
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
CreateTableColumn(5, 64, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: TravelTimeNorms}

sets;
AtomByName([TravelTimeNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TravelTimeNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [TravelTimeNorms], 1, false);
SetAtt([AutoRescaleNorms], 17001);
SetAtt([BlendColors], 0);
SetAtt([ShowOnLayer], 1);
SetAtt([SelectRoute], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(90);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`Level`
0
1
2
3
4
5
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
120
300
600
1200
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
120
300
600
1200
1201
]);
CreateTableColumn(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
]);
CreateTableColumn(5, 64, [`Description`
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(91);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(92);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(93);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(94);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(95);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(96);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(97);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;


{Atom: ProximityNorms}

sets;
AtomByName([ProximityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ProximityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ProximityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(98);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 5);
CreateTableColumn(0, 64, [0
1
2
3
4
5
6
]);
CreateTableColumn(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
]);
CreateTableColumn(2, 98, [`Lower bound`
0
1E-6
1.5
2
2.5
3
]);
CreateTableColumn(3, 102, [`Upper bound`
1E-6
1.5
2
2.5
3
10000000000
]);
CreateTableColumn(4, 64, [`Color`
16777215
8388736
255
33023
65535
65280
]);
CreateTableColumn(5, 105, [`Description`
`empty`
`violation`
`nearby`
`medium`
`far`
`very far`
]);
SetStatus(0);
ExecLateInit;
Up;


{Atom: LstOutputActivityRoutes}

sets;
AtomByName([lstOutputActivityRoutes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstOutputActivityRoutes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [LstOutputActivityRoutes], 1, false);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(99);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: EntryToExit}

sets;
AtomByName([ActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EntryToExit], 1, false);
SetAtt([Interval], 30);
SetAtt([MaxNrIntervals], 40);
SetAtt([BreakDown], 0);
SetAtt([ProfileID], 0);
SetAtt([DensityInterval], 10);
SetAtt([MaxNrDensityIntervals], 20);
SetExprAtt([DensityLowerBound], [{**>= Level E**}
Cell(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([DensityUpperBound], [{**No upper bound**}
0]);
SetAtt([DistanceInterval], 30);
SetAtt([MaxNrDistanceIntervals], 40);
SetAtt([DelayTimeInterval], 10);
SetAtt([MaxNrDelayTimeIntervals], 40);
SetAtt([MaxContentInterval], 60);
SetTextAtt([TravelTimeSettings], []);
SetTextAtt([DistanceSettings], []);
SetTextAtt([DensityPercentageSettings], []);
SetTextAtt([DensityTimeSettings], []);
SetTextAtt([DelayTimeSettings], []);
SetTextAtt([MaxContentSettings], []);
SetAtt([SocialCostInterval], 120);
SetAtt([MaxNrSocialCostIntervals], 40);
SetTextAtt([SocialCostPerAgentSettings], []);
SetTextAtt([SocialCostTotalTimeSettings], []);
SetTextAtt([SocialCostTotalValueSettings], []);
SetAtt([AvgTravelTimePerTimeSettings], 0);
SetAtt([AvgTravelTimePerTimeInterval], 60);
SetAtt([MaxNrProximityTimeIntervals], 40);
SetAtt([ProximityTimeInterval], 10);
SetExprAtt([ProximityLowerBound], [0]);
SetExprAtt([ProximityUpperBound], [{**<= Level A**}
Cell(2, PROXIMITYNORMS_COLUMN_UPPERBOUNDARY, atmProximityNorms)]);
SetTextAtt([ProximityPercentageSettings], []);
SetTextAtt([ProximityTimeSettings], []);
SetAtomSettings([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(100);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 6);
CreateTableColumn(0, 64, [0
1
2
]);
CreateTableColumn(1, 81, [`TravelTimeType`
3331
3331
]);
CreateTableColumn(2, 78, [`ActivityType`
6001
6001
]);
CreateTableColumn(3, 72, [`ActivityGroup`
`-1`
`-1`
]);
CreateTableColumn(4, 64, [`LayerID`
0
0
]);
CreateTableColumn(5, 64, [`ActivityID`
-1
-1
]);
CreateTableColumn(6, 64, [`TimeIn`
0
0
]);
SetStatus(0);
ExecLateInit;


{Atom: TravelTimes}

sets;
AtomByName([TravelTimes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TravelTimes'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TravelTimes], 1, false);
SetAtomSettings([], 0, 1218);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(101);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;
Up;


{Atom: ResultPlayer}

sets;
AtomByName([ResultPlayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ResultPlayer'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ResultPlayer], 1, false);
SetAtt([lstUnusedAgents], 0);
SetAtt([CycleTime], 0);
SetAtt([StartTime], 0);
SetAtt([EndTime], 0);
SetAtt([Initialized], 0);
SetAtt([VisualizeSpeed], 1);
SetAtt([LastTime], 0);
SetAtt([TimeWarpTime], 0);
SetMultiVectorCompareByFields(0, 0);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(102);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 2);
CreateTableColumn(0, 0, [``
1
]);
CreateTableColumn(1, 0, [1
871572480
]);
CreateTableColumn(2, 0, [2
`TransportOutput_TimeWarp`
]);
SetStatus(0);
ExecLateInit;


{Atom: lstUnsedAgents}

sets;
AtomByName([lstUnsedAgents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstUnsedAgents'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstUnsedAgents], 1, false);
SetAtomSettings([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(103);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;
Up;


{Atom: AgentStatistics}

sets;
AtomByName([AgentStatistics], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentStatistics'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentStatistics], 1, false);
SetAtt([OutputStatistics], 0);
SetAtt([OverallDirectory], 0);
SetAtomSettings([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetLoadAtomID(104);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: TransportOutput}

sets;
AtomByName([TransportOutput], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportOutput'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [TransportOutput], 1, false);
SetAtt([CurRow], 0);
SetAtt([GeneratorID], 0);
SetAtomSettings([], 0, 525450);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(105);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
ExecLateInit;


{Atom: TransportOutput_TransportGenerator_1}

sets;
AtomByName([TransportOutput], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TransportOutput'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TransportOutput_TransportGenerator_1], 1, false);
SetAtt([CurRow], 0);
SetAtt([GeneratorID], 1);
SetAtomSettings([], 0, 525450);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetLoadAtomID(106);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(0, 16);
CreateTableColumn(0, 0, [``
]);
CreateTableColumn(1, 0, [1
]);
CreateTableColumn(2, 0, [2
]);
CreateTableColumn(3, 0, [3
]);
CreateTableColumn(4, 0, [4
]);
CreateTableColumn(5, 0, [5
]);
CreateTableColumn(6, 0, [6
]);
CreateTableColumn(7, 0, [7
]);
CreateTableColumn(8, 0, [8
]);
CreateTableColumn(9, 0, [9
]);
CreateTableColumn(10, 0, [10
]);
CreateTableColumn(11, 0, [11
]);
CreateTableColumn(12, 0, [12
]);
CreateTableColumn(13, 0, [13
]);
CreateTableColumn(14, 0, [14
]);
CreateTableColumn(15, 0, [15
]);
CreateTableColumn(16, 0, [16
]);
SetStatus(0);
ExecLateInit;
Up;
Up;
Up;
Up;
LoadEvent(0, 3, 2, 1000000, 0, 0);
LoadEvent(0, 4, 1, 1000001, 0, 0);
LoadEvent(0.362094598035031, 13, 6, 0, 0, 0);
LoadEvent(0, 41, 3, 10000000, 0, 0);
ExecOnCreationOnLoad;
